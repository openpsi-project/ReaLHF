import os
import unittest

import torch
import transformers

from reallm.api.core import model_api
from reallm.impl.model.nn.real_llm_base import PipeCacheData, PipeTransferData, ReaLModel
from reallm.impl.model.nn.real_llm_generate import (generate, GenerationConfig, vanilla_cpu_generate,
                                                    vanilla_packed_generate)
from reallm.impl.model.utils.functional import gather_shifted_log_probs
from tests.utils import *

torch.cuda.manual_seed_all(0)


class ReaLModelStarCoderCPUTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.cuda.set_device(0)
        torch.distributed.init_process_group(
            rank=0,
            world_size=1,
            backend="nccl",
            init_method="tcp://localhost:7778",
        )
        os.environ["LOCAL_RANK"] = str(0)
        import deepspeed

        deepspeed.init_distributed()
        init_global_constants(1, 1, 1)
        cls.bs = bs = 3
        cls.device = device = "cpu"
        cls.dtype = dtype = torch.float32

        sc_cfg = transformers.AutoConfig.from_pretrained("/lustre/meizy/models/starcoder_4l/")
        sc_cfg.n_layer = 2
        sc_cfg.n_embd = 1024
        sc_cfg.n_head = 8
        sc_cfg.n_inner = 4096
        sc_cfg.n_positions = 512

        cls.tokenizer = model_api.load_hf_tokenizer("/lustre/meizy/models/starcoder_4l/")
        cls.tokenizer.pad_token_id = cls.tokenizer.eos_token_id

        cls.starcoder: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_config(
            sc_cfg).to(dtype=dtype, device=device)
        cls.starcoder.eval()

        cls.model = ReaLModel.from_starcoder(from_model=cls.starcoder, dtype=dtype, device=device)
        cls.model.eval()
        cls.config = cls.model.config

    @torch.no_grad()
    def testStandardForward(self):
        config = self.config
        bs = self.bs
        starcoder = self.starcoder
        model = self.model
        device = self.device

        max_seq_len = 6
        input_ids = torch.randint(0, config.vocab_size, (bs, max_seq_len), dtype=torch.long, device=device)
        seqlens = torch.randint(3, max_seq_len, (bs,), dtype=torch.long, device=device)
        seqlens[0] = max_seq_len
        leftpad_attention_mask = torch.ones((bs, max_seq_len), dtype=torch.bool, device=device)
        rightpad_attention_mask = torch.ones((bs, max_seq_len), dtype=torch.bool, device=device)
        for i in range(bs):
            leftpad_attention_mask[i, :max_seq_len - seqlens[i]] = 0
            rightpad_attention_mask[i, seqlens[i]:] = 0

        # no mask
        x = PipeTransferData()
        ys = [PipeCacheData(packed_input_ids=input_ids)
              ] + [PipeCacheData() for _ in range(config.n_layers + 1)]
        logits = model(x, ys).pp_output
        sc_logits = starcoder(input_ids=input_ids).logits
        assert torch.allclose(logits, sc_logits, atol=2e-5), ((logits - sc_logits)).abs().max()

        # right pad
        x = PipeTransferData(attention_mask=rightpad_attention_mask)
        ys = [PipeCacheData(packed_input_ids=input_ids)
              ] + [PipeCacheData() for _ in range(config.n_layers + 1)]
        logits = model(x, ys).pp_output
        sc_logits = starcoder(input_ids=input_ids, attention_mask=rightpad_attention_mask).logits
        assert torch.allclose(logits, sc_logits, atol=2e-5), ((logits - sc_logits)).abs().max()

        # left pad
        x = PipeTransferData(attention_mask=leftpad_attention_mask)
        ys = [PipeCacheData(packed_input_ids=input_ids)
              ] + [PipeCacheData() for _ in range(config.n_layers + 1)]
        logits = model(x, ys).pp_output
        sc_logits = starcoder(input_ids=input_ids, attention_mask=leftpad_attention_mask).logits
        assert torch.allclose(logits, sc_logits, atol=2e-5), ((logits - sc_logits)).abs().max()

    @torch.no_grad()
    def testGenerate(self):
        seqs = [
            "# This is a print function\ndef",
            "import time\n",
            "assert torch.allclose(logits, sc_logits, atol=5e-3",
        ]
        self.tokenizer.padding_side = "left"
        encoding = self.tokenizer(seqs, return_tensors="pt", padding=True)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        gconfig = GenerationConfig(
            min_new_tokens=0,
            max_new_tokens=100,
            temperature=1.0,
            greedy=True,
            top_k=50,
            top_p=1.0,
            num_samples=1,
        )
        vg, vglogprob_, vglmask = vanilla_cpu_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            gconfig=gconfig,
        )
        self.assertIsNone(vglmask)

        vg_input_ids = torch.cat([input_ids, vg], -1)
        vam = torch.logical_and(
            vg_input_ids.not_equal(self.tokenizer.pad_token_id),
            (vg_input_ids.not_equal(self.tokenizer.eos_token_id)),
        ).long()

        x = PipeTransferData(attention_mask=vam)
        ys = [PipeCacheData(packed_input_ids=vg_input_ids)
              ] + [PipeCacheData() for _ in range(self.model.config.n_layers + 1)]
        vglogits = self.model(x, ys).pp_output
        vglogprob = gather_shifted_log_probs(vglogits, vg_input_ids)
        vglogprob = vglogprob[:, input_ids.shape[1] - 1:]
        assert torch.allclose(vglogprob_, vglogprob)

        tgconfig = transformers.GenerationConfig(
            min_new_tokens=0,
            max_new_tokens=100,
            temperature=1.0,
            do_sample=False,
            top_k=50,
            top_p=1.0,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        tseq = self.starcoder.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=tgconfig,
        )
        tam = torch.logical_and(tseq.not_equal(self.tokenizer.pad_token_id),
                                (tseq.not_equal(self.tokenizer.eos_token_id))).long()
        tlogits = self.starcoder(input_ids=tseq, attention_mask=tam).logits.float()
        tlogprob = gather_shifted_log_probs(tlogits, tseq)
        tseq = tseq[:, input_ids.shape[1]:]
        tlogprob = tlogprob[:, input_ids.shape[1] - 1:]
        assert torch.allclose(tseq, vg)
        assert torch.allclose(vglogprob, tlogprob), (vglogprob - tlogprob).abs().max()


class ReaLModelStarCoderTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.random.manual_seed(0)
        cls.bs = bs = 4
        cls.device = device = "cuda"

        sc_cfg = transformers.AutoConfig.from_pretrained("/lustre/meizy/models/starcoder_4l/")
        sc_cfg.n_layer = 1
        sc_cfg.n_embd = 1024
        sc_cfg.n_head = 8
        sc_cfg.n_inner = 4096
        sc_cfg.n_positions = 512

        cls.tokenizer = model_api.load_hf_tokenizer("/lustre/meizy/models/starcoder_4l/")
        cls.tokenizer.pad_token_id = cls.tokenizer.eos_token_id

        cls.starcoder: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_config(
            sc_cfg).to(dtype=torch.float16, device=device)
        cls.starcoder.eval()

        cls.model = ReaLModel.from_starcoder(from_model=cls.starcoder, dtype=torch.float16, device=device)
        cls.model.eval()
        cls.config = cls.model.config

    def testStandardForward(self):
        config = self.config
        bs = self.bs
        starcoder = self.starcoder
        model = self.model
        device = self.device

        input_ids = torch.randint(0,
                                  config.vocab_size, (bs, config.n_positions),
                                  dtype=torch.long,
                                  device=device)

        x = PipeTransferData()
        ys = [PipeCacheData(packed_input_ids=input_ids)
              ] + [PipeCacheData() for _ in range(config.n_layers + 1)]
        with torch.no_grad():
            logits = model(x, ys).pp_output
            sc_logits = starcoder(input_ids=input_ids).logits
        assert torch.allclose(logits, sc_logits, atol=5e-3), ((logits - sc_logits)).abs().max()

    def testPackedForward(self):
        config = self.config
        bs = self.bs
        starcoder = self.starcoder
        model = self.model
        device = self.device

        input_ids = torch.randint(0,
                                  config.vocab_size, (bs, config.n_positions),
                                  dtype=torch.long,
                                  device=device)
        input_len = torch.randint(10, config.n_positions, (bs,), dtype=torch.long, device=device)
        attention_mask = torch.ones(bs, config.n_positions, dtype=torch.bool, device=device)
        for i in range(bs):
            attention_mask[i, input_len[i]:] = False
        packed_input_ids = torch.cat([input_ids[i, :input_len[i]] for i in range(bs)])
        cu_seqlens = torch.cat(
            [torch.tensor([0], dtype=torch.long, device=device),
             torch.cumsum(input_len, dim=0)]).to(torch.int32)
        max_seqlen = int(input_len.max().item())
        total_seqlen = input_len.sum()

        with torch.no_grad():
            sc_output = starcoder(input_ids=input_ids, attention_mask=attention_mask).logits
            # sc_logits = sc_output.flatten(end_dim=1)[:-100]
            sc_logits = torch.zeros(total_seqlen, config.vocab_size, dtype=torch.float16, device=device)
            for i in range(bs):
                sc_logits[cu_seqlens[i]:cu_seqlens[i + 1]] = sc_output[i, :input_len[i]]

        x = PipeTransferData(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        ys = [PipeCacheData(packed_input_ids=packed_input_ids)
              ] + [PipeCacheData() for _ in range(config.n_layers + 1)]
        with torch.no_grad():
            logits = model(x, ys).pp_output
        assert torch.allclose(logits, sc_logits, atol=5e-3), ((logits - sc_logits)).abs().max()

    def testGenerate(self):
        seqs = [
            "# This is a print function\ndef",
            "import time\n",
            "assert torch.allclose(logits, sc_logits, atol=5e-3",
        ]
        self.tokenizer.padding_side = "left"
        encoding = self.tokenizer(seqs, return_tensors="pt", padding=True)
        input_ids = encoding["input_ids"].to(self.device)
        prompt_len = input_ids.shape[1]
        attention_mask = encoding["attention_mask"].to(self.device)
        gconfig = GenerationConfig(
            min_new_tokens=0,
            max_new_tokens=100,
            temperature=1.0,
            greedy=True,
            top_k=50,
            top_p=1.0,
            num_samples=1,
        )
        generated, glogprobs, glmask, _, _ = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            gconfig=gconfig,
        )
        self.assertIsNone(glmask)
        tgconfig = transformers.GenerationConfig(
            min_new_tokens=0,
            max_new_tokens=100,
            temperature=1.0,
            do_sample=False,
            top_k=50,
            top_p=1.0,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        with torch.no_grad():
            tseq = self.starcoder.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=tgconfig,
            )[:, prompt_len:]
        assert torch.allclose(tseq, generated), (tseq, generated)

        inf_input_ids = torch.cat([input_ids, generated], -1)
        tam = torch.logical_and(
            inf_input_ids.not_equal(self.tokenizer.pad_token_id),
            (inf_input_ids.not_equal(self.tokenizer.eos_token_id)),
        ).long()
        tlogits = self.starcoder(input_ids=inf_input_ids, attention_mask=tam).logits.float()
        tlogprobs = gather_shifted_log_probs(tlogits, inf_input_ids)
        tlogprobs = tlogprobs[:, prompt_len - 1:]
        assert torch.allclose(glogprobs, tlogprobs, atol=5e-3), (glogprobs - tlogprobs).abs().max()

    def testGenerateFromCache(self):
        seqs = [
            "# This is a print function\ndef",
            "import time\n",
            "assert torch.allclose(logits, sc_logits, atol=5e-3",
        ]
        self.tokenizer.padding_side = "left"
        encoding = self.tokenizer(seqs, return_tensors="pt", padding=True)
        prompts = encoding["input_ids"].to(self.device)
        prompt_att_mask = encoding["attention_mask"].to(self.device)
        prompt_len = prompts.shape[1]

        origin_max_new_tokens = 100
        gconfig = GenerationConfig(
            min_new_tokens=1,
            max_new_tokens=origin_max_new_tokens,
            temperature=1.0,
            greedy=True,
            top_k=50,
            top_p=1.0,
            num_samples=1,
        )
        seq, log_probs, logits_mask, ys, _ = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=prompts,
            attention_mask=prompt_att_mask,
            gconfig=gconfig,
        )
        original_k_caches = [y.k_cache.clone() for y in ys]
        original_v_caches = [y.v_cache.clone() for y in ys]

        first_n_tokens = 50 - prompt_len
        gconfig.max_new_tokens = first_n_tokens
        seq21, log_probs21, logits_mask21, tmp_ys, _ = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=prompts,
            attention_mask=prompt_att_mask,
            gconfig=gconfig,
        )
        k_caches = [y.k_cache for y in tmp_ys]
        v_caches = [y.v_cache for y in tmp_ys]
        cache_seqlens = tmp_ys[0].cache_seqlens
        gconfig.min_new_tokens = 0
        gconfig.max_new_tokens = origin_max_new_tokens - first_n_tokens
        seq22, log_probs22, logits_mask22, ys2, _ = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=seq21[:, -1:],
            k_caches=k_caches,
            v_caches=v_caches,
            cache_seqlens=cache_seqlens,
            gconfig=gconfig,
        )
        self.assertIsNone(logits_mask22)
        seq2 = torch.cat([seq21, seq22], -1)
        log_probs2 = torch.cat([log_probs21, log_probs22], -1)
        logits_mask2 = torch.cat(
            [
                logits_mask21,
                torch.ones(
                    (len(seqs), origin_max_new_tokens - first_n_tokens, self.config.vocab_size),
                    dtype=torch.bool,
                    device=self.device,
                ),
            ],
            -2,
        )

        assert torch.allclose(seq2, seq), (seq, seq21, seq22)
        assert torch.allclose(log_probs, log_probs2)
        assert torch.allclose(logits_mask[:, :first_n_tokens], logits_mask21)
        assert torch.allclose(logits_mask, logits_mask2)
        for (k_cache, v_cache), y2 in zip(zip(original_k_caches, original_v_caches), ys2):
            assert torch.allclose(k_cache, y2.k_cache)
            assert torch.allclose(v_cache, y2.v_cache)

    def testGenerateTwoOrMoreSamples(self):
        seqs = [
            "# This is a print function\ndef",
            "import time\n",
            "assert torch.allclose(logits, sc_logits, atol=5e-3",
            "I'm really happy about",
        ]
        num_samples = 3
        self.tokenizer.padding_side = "left"
        encoding = self.tokenizer(seqs, return_tensors="pt", padding=True)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        for greedy in [True, False]:
            for min_new_tokens in [0, 1]:
                gconfig = GenerationConfig(
                    min_new_tokens=min_new_tokens,
                    max_new_tokens=10,
                    temperature=1.0,
                    greedy=greedy,
                    top_k=50,
                    top_p=1.0,
                    num_samples=num_samples,
                )
                generated, glogprobs, glmask, _, _ = generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    gconfig=gconfig,
                )
                if greedy and min_new_tokens == 0:
                    self.assertIsNone(glmask)
                else:
                    self.assertEqual(glmask.shape[0], num_samples * len(seqs))
                self.assertEqual(generated.shape[0], num_samples * len(seqs))
                self.assertEqual(glogprobs.shape[0], num_samples * len(seqs))


if __name__ == "__main__":
    unittest.main()
