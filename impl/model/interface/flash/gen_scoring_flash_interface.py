from typing import Dict, List, Optional
import dataclasses
import json
import os

import deepspeed
import torch
import torch.distributed as dist
import transformers

from base.namedarray import NamedArray, recursive_apply
from impl.model.nn.flash_mqat.flash_generate import GenerationConfig
from impl.model.nn.flash_mqat.flash_mqat_interface import HuggingfaceLikeFlashMQATForCausalLM
import api.huggingface
import api.model


@dataclasses.dataclass
class PackedGenScoringInterface(api.model.ModelInterface):
    generation_config: Optional[Dict] = None

    def __post_init__(self):
        self.score_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            "/lustre/fw/pretrained/distilbert-base-uncased-finetuned-sst-2-english").cuda()
        self.score_tokenizer = api.huggingface.load_hf_tokenizer(
            "/lustre/fw/pretrained/distilbert-base-uncased-finetuned-sst-2-english")

        self.sft_model = HuggingfaceLikeFlashMQATForCausalLM.from_pretrained(
            model_path=
            "/data/aigc/llm/checkpoints/fw/flash-ppo-s42/run20231106-rw2/actor@pp_00-mp_00-dp_00/epoch0step0/",
            dtype=torch.float16,
            device="cuda",
        )
        self.history_data = []

        self.score_model.eval()
        self.sft_model.eval()

    def save(self, model: api.model.Model, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "gen_score_data.jsonl"), "w") as f:
            for x in self.history_data:
                json.dump(x, f)
                f.write("\n")

    @torch.inference_mode()
    def generate(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        module = model.module
        if isinstance(module, deepspeed.DeepSpeedEngine):
            # we don't calculate gradient here, so it's safe to unwrap deepspeed
            module = module.module

        module.eval()
        assert isinstance(module, HuggingfaceLikeFlashMQATForCausalLM)

        gconfig = GenerationConfig(**self.generation_config)
        num_samples = gconfig.num_samples

        data = recursive_apply(data, lambda x: x.to(model.device))

        prompts: torch.LongTensor = data["prompts"]

        all_prompts = [torch.zeros_like(prompts) for _ in range(dist.get_world_size())]
        dist.all_gather(all_prompts, prompts)

        prompt_att_mask: torch.BoolTensor = data["prompt_att_mask"]
        bs, prompt_max_len = prompts.shape[:2]
        gen_res = module.generate(
            tokenizer=model.tokenizer,
            input_ids=prompts,
            attention_mask=prompt_att_mask,
            gconfig=gconfig,
        )
        gen_tokens = gen_res.sequences
        logp = gen_res.scores.float()
        logits_mask = gen_res.logits_mask

        eos_token_id = model.tokenizer.eos_token_id
        pad_token_id = model.tokenizer.pad_token_id
        non_pad_eos_mask = (gen_tokens != pad_token_id).logical_and(gen_tokens != eos_token_id)
        gen_lengths = non_pad_eos_mask.sum(dim=-1) + 1
        gen_lengths = gen_lengths.clip(max=gen_tokens.shape[-1])
        if eos_token_id == pad_token_id:
            eos_indices = non_pad_eos_mask.shape[1] - torch.argmax(
                (non_pad_eos_mask.flip(1) != 0).float(), dim=1)
            eos_indices.clip_(max=non_pad_eos_mask.shape[1] - 1)
            seq_att_mask = non_pad_eos_mask.clone()
            batch_indices = torch.arange(seq_att_mask.shape[0], device=model.device, dtype=torch.long)
            seq_att_mask[batch_indices, eos_indices] = 1
        else:
            seq_att_mask = gen_tokens != pad_token_id

        prompts = prompts.unsqueeze(1).repeat(1, num_samples, 1).flatten(end_dim=1)
        prompt_att_mask = prompt_att_mask.unsqueeze(1).repeat(1, num_samples, 1).flatten(end_dim=1)
        seq = torch.cat([prompts, gen_tokens], dim=1)
        prompt_lens = prompt_att_mask.sum(1)
        attention_mask = torch.cat([prompt_att_mask, seq_att_mask], 1)

        # Compute the log probability outputed by the SFT model
        sft_model_logits = self.sft_model(input_ids=seq, attention_mask=attention_mask,
                                          padding_side=None).logits
        sft_seqlogp = []
        for i in range(prompts.shape[0]):
            logits_ = sft_model_logits[i][prompt_lens[i] - 1:-1]
            if not gconfig.greedy:
                logits_ /= gconfig.temperature
                m_ = logits_mask[i, :gen_lengths[i]]
                logits_.masked_fill_(m_.logical_not(), torch.finfo(logits_.dtype).min)
            log_probs_ = torch.nn.functional.log_softmax(logits_, dim=-1)
            sslp = torch.gather(log_probs_, -1, gen_tokens[i, :gen_lengths[i]].unsqueeze(-1)).squeeze(-1)
            sft_seqlogp.append(sslp.sum())
        sft_seqlogp = torch.stack(sft_seqlogp)

        seqlogp = []
        for i in range(logp.shape[0]):
            seqlogp.append(logp[i, :gen_lengths[i]].sum())
        seqlogp = torch.stack(seqlogp)

        all_seqlogp = [torch.zeros_like(seqlogp) for _ in range(dist.get_world_size())]
        all_sft_seqlogp = [torch.zeros_like(sft_seqlogp) for _ in range(dist.get_world_size())]
        dist.all_gather(all_seqlogp, seqlogp)
        dist.all_gather(all_sft_seqlogp, sft_seqlogp)

        if seq.shape[1] < gconfig.max_new_tokens + prompt_max_len:
            seq = torch.nn.functional.pad(
                seq,
                (0, gconfig.max_new_tokens + prompt_max_len - seq.shape[1]),
                value=model.tokenizer.pad_token_id,
            )

        all_seq = [torch.zeros_like(seq) for _ in range(dist.get_world_size())]
        dist.all_gather(all_seq, seq)

        texts = model.tokenizer.batch_decode(seq, skip_special_tokens=True)

        # pipe = transformers.pipeline("text-classification",
        #                              model=self.score_model,
        #                              tokenizer=self.score_tokenizer)
        # # a list of dict which contains 'label' and 'score', 'label' can be 'NEGATIVE' or 'POSITIVE' and 'score' is the probability
        # senti_res: List[Dict] = pipe(texts)

        score_encoding = self.score_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        logits = self.score_model(
            input_ids=score_encoding["input_ids"].cuda(),
            attention_mask=score_encoding["attention_mask"].cuda(),
        ).logits.float()
        # For IMDB, 0 is negative and 1 is positive. We record the probability of positive.
        probs = torch.softmax(logits, dim=-1)
        score = probs[..., -1].contiguous()

        all_score = [torch.zeros_like(score) for _ in range(dist.get_world_size())]
        dist.all_gather(all_score, score)

        prompts_str = model.tokenizer.batch_decode(torch.cat(all_prompts, 0), skip_special_tokens=True)
        assert len(prompts_str) == bs * dist.get_world_size()
        texts = model.tokenizer.batch_decode(torch.cat(all_seq, 0), skip_special_tokens=True)
        score = torch.cat(all_score, 0)
        seqlogp = torch.cat(all_seqlogp, 0)
        sft_seqlogp = torch.cat(all_sft_seqlogp, 0)
        assert len(texts) == bs * num_samples * dist.get_world_size() == score.shape[0] == seqlogp.shape[0]

        for i in range(bs * dist.get_world_size()):
            prompt = prompts_str[i]
            prompt_answers = texts[i * num_samples:(i + 1) * num_samples]
            try:
                for j, a in enumerate(prompt_answers):
                    assert a.startswith(prompt), (a, prompt)
                    prompt_answers[j] = a[len(prompt):]
            except AssertionError:
                continue
            lp = seqlogp[i * num_samples:(i + 1) * num_samples]
            slp = sft_seqlogp[i * num_samples:(i + 1) * num_samples]
            s = score[i * num_samples:(i + 1) * num_samples]
            x = dict(
                prompt=prompt,
                answers=prompt_answers,
                scores=s.cpu().tolist(),
                seqlogp=lp.cpu().tolist(),
                kl=(lp - slp).cpu().tolist(),
            )
            self.history_data.append(x)

        return {}


api.model.register_interface("flash_gen_score", PackedGenScoringInterface)
