from typing import Dict, List, Optional
import dataclasses
import json
import os

import deepspeed
import torch
import torch.distributed as dist
import transformers

from base.namedarray import NamedArray, recursive_apply
import api.huggingface
import api.model
import impl.model.nn.flash_mqat as flash_mqat


@dataclasses.dataclass
class PackedGenScoringInterface(api.model.ModelInterface):
    generation_config: Optional[Dict] = None

    def __post_init__(self):
        self.score_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            "/lustre/fw/pretrained/distilbert-base-uncased-finetuned-sst-2-english").cuda()
        self.score_tokenizer = api.huggingface.load_hf_tokenizer(
            "/lustre/fw/pretrained/distilbert-base-uncased-finetuned-sst-2-english")
        self.history_data = []

    def save(self, model: api.model.Model, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "gen_score_data.jsonl"), 'w') as f:
            for x in self.history_data:
                json.dump(x, f)
                f.write('\n')

    @torch.inference_mode()
    def generate(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        module = model.module
        if isinstance(module, deepspeed.DeepSpeedEngine):
            # we don't calculate gradient here, so it's safe to unwrap deepspeed
            module = module.module

        module.eval()
        assert isinstance(module, flash_mqat.HuggingfaceLikeFlashMQATForCausalLM)

        gconfig = flash_mqat.GenerationConfig(**self.generation_config)
        num_samples = gconfig.num_samples

        data = recursive_apply(data, lambda x: x.to(model.device))

        prompts: torch.LongTensor = data['prompts']

        all_prompts = [torch.zeros_like(prompts) for _ in range(dist.get_world_size())]
        dist.all_gather(all_prompts, prompts)

        prompt_att_mask: torch.BoolTensor = data['prompt_att_mask']
        bs, prompt_max_len = prompts.shape[:2]
        gen_tokens = module.generate(
            tokenizer=model.tokenizer,
            input_ids=prompts,
            attention_mask=prompt_att_mask,
            gconfig=gconfig,
        ).sequences

        prompts = prompts.unsqueeze(1).repeat(1, num_samples, 1).flatten(end_dim=1)
        seq = torch.cat([prompts, gen_tokens], dim=1)

        if seq.shape[1] < gconfig.max_new_tokens + prompt_max_len:
            seq = torch.nn.functional.pad(seq, (0, gconfig.max_new_tokens + prompt_max_len - seq.shape[1]),
                                          value=model.tokenizer.pad_token_id)

        all_seq = [torch.zeros_like(seq) for _ in range(dist.get_world_size())]
        dist.all_gather(all_seq, seq)

        texts = model.tokenizer.batch_decode(seq, skip_special_tokens=True)

        # pipe = transformers.pipeline("text-classification",
        #                              model=self.score_model,
        #                              tokenizer=self.score_tokenizer)
        # # a list of dict which contains 'label' and 'score', 'label' can be 'NEGATIVE' or 'POSITIVE' and 'score' is the probability
        # senti_res: List[Dict] = pipe(texts)

        score_encoding = self.score_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        logits = self.score_model(input_ids=score_encoding['input_ids'].cuda(),
                                  attention_mask=score_encoding['attention_mask'].cuda()).logits.float()
        # For IMDB, 0 is negative and 1 is positive. We record the probability of positive.
        probs = torch.softmax(logits, dim=-1)
        score = probs[..., -1].contiguous()

        all_score = [torch.zeros_like(score) for _ in range(dist.get_world_size())]
        dist.all_gather(all_score, score)

        prompts_str = model.tokenizer.batch_decode(torch.cat(all_prompts, 0), skip_special_tokens=True)
        assert len(prompts_str) == bs * dist.get_world_size()
        texts = model.tokenizer.batch_decode(torch.cat(all_seq, 0), skip_special_tokens=True)
        score = torch.cat(all_score, 0)
        assert len(texts) == bs * num_samples * dist.get_world_size() == score.shape[0]

        for i in range(bs * dist.get_world_size()):
            prompt = prompts_str[i]
            prompt_answers = texts[i * num_samples:(i + 1) * num_samples]
            try:
                for j, a in enumerate(prompt_answers):
                    assert a.startswith(prompt), (a, prompt)
                    prompt_answers[j] = a[len(prompt):]
            except AssertionError:
                continue
            s = score[i * num_samples:(i + 1) * num_samples]
            x = dict(prompt=prompt, answers=prompt_answers, scores=s.cpu().tolist())
            self.history_data.append(x)

        return {}


api.model.register_interface('flash_gen_score', PackedGenScoringInterface)