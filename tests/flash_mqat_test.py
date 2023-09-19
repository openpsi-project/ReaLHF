import unittest

import torch
import transformers

from impl.model.nn.flash_mqat import FlashMQATForCausalLM, PipeCacheData, PipeTransferData
from impl.model.utils.flash_generate import (build_packed_inputs, generate, GenerationConfig, unpack_tensor,
                                             vanilla_cpu_generate, vanilla_packed_generate)


class FlashMQATStarCoderTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.bs = bs = 4
        cls.device = device = 'cuda'

        sc_cfg = transformers.AutoConfig.from_pretrained("/data/aigc/llm/checkpoints/4l-starcoder/")
        sc_cfg.n_layer = 1
        sc_cfg.n_embd = 1024
        sc_cfg.n_head = 8
        sc_cfg.n_inner = 4096
        sc_cfg.n_positions = 512

        cls.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "/hddlustre/llm/public/checkpoints/pretrained/starcoder/")
        cls.tokenizer.pad_token_id = cls.tokenizer.eos_token_id

        cls.starcoder: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_config(
            sc_cfg).to(dtype=torch.float16, device=device)
        cls.starcoder.eval()

        cls.model = FlashMQATForCausalLM.from_starcoder(from_model=cls.starcoder,
                                                        dtype=torch.float16,
                                                        device=device)
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
        ys = [PipeCacheData(input_ids=input_ids)] + [PipeCacheData() for _ in range(config.n_layers + 1)]
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
             torch.cumsum(input_len, dim=0)])
        max_seqlen = int(input_len.max().item())
        total_seqlen = input_len.sum()

        with torch.no_grad():
            sc_output = starcoder(input_ids=input_ids, attention_mask=attention_mask).logits
            # sc_logits = sc_output.flatten(end_dim=1)[:-100]
            sc_logits = torch.zeros(total_seqlen, config.vocab_size, dtype=torch.float16, device=device)
            for i in range(bs):
                sc_logits[cu_seqlens[i]:cu_seqlens[i + 1]] = sc_output[i, :input_len[i]]

        x = PipeTransferData(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        ys = [PipeCacheData(input_ids=packed_input_ids)
              ] + [PipeCacheData() for _ in range(config.n_layers + 1)]
        with torch.no_grad():
            logits = model(x, ys).pp_output
        assert torch.allclose(logits, sc_logits, atol=5e-3), ((logits - sc_logits)).abs().max()

    def testGenerate(self):
        seqs = [
            "# This is a print function\ndef", "import time\n",
            "assert torch.allclose(logits, sc_logits, atol=5e-3"
        ]
        self.tokenizer.padding_side = "left"
        encoding = self.tokenizer(seqs, return_tensors="pt", padding=True)
        input_ids = encoding['input_ids'].to(self.device)
        prompt_len = input_ids.shape[1]
        attention_mask = encoding['attention_mask'].to(self.device)
        gconfig = GenerationConfig(min_new_tokens=10,
                                   max_new_tokens=100,
                                   temperature=1.0,
                                   greedy=True,
                                   top_k=50,
                                   top_p=1.0,
                                   num_samples=1)
        generated, glogits, glmask = generate(model=self.model,
                                              tokenizer=self.tokenizer,
                                              input_ids=input_ids,
                                              attention_mask=attention_mask,
                                              gconfig=gconfig)
        tgconfig = transformers.GenerationConfig(
            min_new_tokens=10,
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
        tam = torch.logical_and(inf_input_ids.not_equal(self.tokenizer.pad_token_id),
                                (inf_input_ids.not_equal(self.tokenizer.eos_token_id))).long()
        tlogits = self.starcoder(input_ids=inf_input_ids, attention_mask=tam).logits[:, :-1]
        tlogits = tlogits.gather(-1, inf_input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        tlogits = tlogits[:, prompt_len - 1:]
        assert torch.allclose(glogits, tlogits, atol=5e-3), (glogits - tlogits).abs().max()


class FlashMQATStarCoderCPUTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.bs = bs = 3
        cls.device = device = 'cpu'
        cls.dtype = dtype = torch.float32

        sc_cfg = transformers.AutoConfig.from_pretrained("/data/aigc/llm/checkpoints/4l-starcoder/")
        sc_cfg.n_layer = 2
        sc_cfg.n_embd = 1024
        sc_cfg.n_head = 8
        sc_cfg.n_inner = 4096
        sc_cfg.n_positions = 512

        cls.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "/hddlustre/llm/public/checkpoints/pretrained/starcoder/")
        cls.tokenizer.pad_token_id = cls.tokenizer.eos_token_id

        cls.starcoder: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_config(
            sc_cfg).to(dtype=dtype, device=device)
        cls.starcoder.eval()

        cls.model = FlashMQATForCausalLM.from_starcoder(from_model=cls.starcoder, dtype=dtype, device=device)
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
        ys = [PipeCacheData(input_ids=input_ids)] + [PipeCacheData() for _ in range(config.n_layers + 1)]
        logits = model(x, ys).pp_output
        sc_logits = starcoder(input_ids=input_ids).logits
        assert torch.allclose(logits, sc_logits, atol=2e-5), ((logits - sc_logits)).abs().max()

        # right pad
        x = PipeTransferData(attention_mask=rightpad_attention_mask)
        ys = [PipeCacheData(input_ids=input_ids)] + [PipeCacheData() for _ in range(config.n_layers + 1)]
        logits = model(x, ys).pp_output
        sc_logits = starcoder(input_ids=input_ids, attention_mask=rightpad_attention_mask).logits
        assert torch.allclose(logits, sc_logits, atol=2e-5), ((logits - sc_logits)).abs().max()

        # left pad
        x = PipeTransferData(attention_mask=leftpad_attention_mask)
        ys = [PipeCacheData(input_ids=input_ids)] + [PipeCacheData() for _ in range(config.n_layers + 1)]
        logits = model(x, ys).pp_output
        sc_logits = starcoder(input_ids=input_ids, attention_mask=leftpad_attention_mask).logits
        assert torch.allclose(logits, sc_logits, atol=2e-5), ((logits - sc_logits)).abs().max()

    @torch.no_grad()
    def testGenerate(self):
        seqs = [
            "# This is a print function\ndef", "import time\n",
            "assert torch.allclose(logits, sc_logits, atol=5e-3"
        ]
        self.tokenizer.padding_side = "left"
        encoding = self.tokenizer(seqs, return_tensors="pt", padding=True)
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        gconfig = GenerationConfig(min_new_tokens=10,
                                   max_new_tokens=100,
                                   temperature=1.0,
                                   greedy=True,
                                   top_k=50,
                                   top_p=1.0,
                                   num_samples=1)
        vg, vglogits_, vglmask = vanilla_cpu_generate(model=self.model,
                                                      tokenizer=self.tokenizer,
                                                      input_ids=input_ids,
                                                      attention_mask=attention_mask,
                                                      gconfig=gconfig)
        vg_input_ids = torch.cat([input_ids, vg], -1)
        vam = torch.logical_and(vg_input_ids.not_equal(self.tokenizer.pad_token_id),
                                (vg_input_ids.not_equal(self.tokenizer.eos_token_id))).long()
        x = PipeTransferData(attention_mask=vam)
        ys = [PipeCacheData(input_ids=vg_input_ids)
              ] + [PipeCacheData() for _ in range(self.model.config.n_layers + 1)]
        vglogits = self.model(x, ys).pp_output[:, :-1]
        vglogits = vglogits.gather(-1, vg_input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)[:,
                                                                                      input_ids.shape[1] - 1:]
        assert torch.allclose(vglogits_, vglogits)

        tgconfig = transformers.GenerationConfig(
            min_new_tokens=10,
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
        tlogits = self.starcoder(input_ids=tseq, attention_mask=tam).logits[:, :-1]
        tlogits = tlogits.gather(-1, tseq[:, 1:][..., None]).squeeze(-1)
        tseq = tseq[:, input_ids.shape[1]:]
        tlogits = tlogits[:, input_ids.shape[1] - 1:]
        assert torch.allclose(tseq, vg)
        assert torch.allclose(vglogits, tlogits), (vglogits - tlogits).abs().max()


class FlashMQATCPUGPUAccordanceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.bs = bs = 7
        cls.device = device = 'cpu'
        cls.dtype = dtype = torch.float32

        sc_cfg = transformers.AutoConfig.from_pretrained("/data/aigc/llm/checkpoints/4l-starcoder/")
        sc_cfg.n_layer = 1
        sc_cfg.n_embd = 1024
        sc_cfg.n_head = 8
        sc_cfg.n_inner = 4096
        sc_cfg.n_positions = 512

        cls.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "/hddlustre/llm/public/checkpoints/pretrained/starcoder/")
        cls.tokenizer.pad_token_id = cls.tokenizer.eos_token_id

        starcoder: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_config(sc_cfg).to(
            dtype=dtype, device=device)
        starcoder.eval()

        cls.model = FlashMQATForCausalLM.from_starcoder(from_model=starcoder, dtype=dtype, device=device)
        cls.model.eval()
        cls.config = cls.model.config

    @torch.no_grad()
    def testGenerate(self):
        seqs = [
            "# This is a print function\ndef", "import time\n",
            "assert torch.allclose(logits, sc_logits, atol=5e-3"
        ]
        self.tokenizer.padding_side = "left"
        encoding = self.tokenizer(seqs, return_tensors="pt", padding=True)
        prompt: torch.Tensor = encoding['input_ids'].to(self.device)
        prompt_att_mask: torch.Tensor = encoding['attention_mask'].to(self.device)
        gconfig = GenerationConfig(min_new_tokens=10,
                                   max_new_tokens=100,
                                   temperature=1.0,
                                   greedy=True,
                                   top_k=50,
                                   top_p=1.0,
                                   num_samples=1)
        vcg, vclogits, vcmask = vanilla_cpu_generate(model=self.model,
                                                     tokenizer=self.tokenizer,
                                                     input_ids=prompt,
                                                     attention_mask=prompt_att_mask,
                                                     gconfig=gconfig)

        self.model = self.model.cuda().to(torch.float16)
        vg, vglogits, vgmask = vanilla_packed_generate(model=self.model.cuda(),
                                                       tokenizer=self.tokenizer,
                                                       input_ids=prompt.cuda(),
                                                       attention_mask=prompt_att_mask.cuda(),
                                                       gconfig=gconfig)
        vglogits = vglogits.float().cpu()
        vgmask = vgmask.float().cpu()

        seq = torch.cat([prompt, vcg], -1)
        seq_attn_mask = torch.logical_and(seq.ne(self.tokenizer.pad_token_id),
                                          seq.ne(self.tokenizer.eos_token_id))
        packed_input_ids, cu_seqlens, max_seq_len = build_packed_inputs(seq, seq_attn_mask, 'cuda')
        x = PipeTransferData(cu_seqlens=cu_seqlens.cuda(), max_seqlen=max_seq_len)
        ys = [PipeCacheData(input_ids=packed_input_ids.cuda())
              ] + [PipeCacheData() for _ in range(self.model.config.n_layers + 1)]
        inf_logits = self.model(x, ys).pp_output[:, :-1].float().cpu()
        inf_logits = unpack_tensor(inf_logits, cu_seqlens, 'cpu', padding_side='left')
        inf_logits = inf_logits.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)[:, prompt.shape[1] - 1:]
        assert torch.allclose(vcg, vg.cpu()), (vcg, vg)
        assert torch.allclose(inf_logits, vclogits,
                              atol=5e-3), (inf_logits, vclogits, (inf_logits - vclogits).abs().max())
        assert torch.allclose(vglogits, vclogits,
                              atol=5e-3), (vglogits, vclogits, (vglogits - vclogits).abs().max())
        assert torch.allclose(vgmask.cpu(), vcmask), (vgmask, vcmask)


class FlashMQATGPUGPUAccordanceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.bs = bs = 7
        cls.device = device = 'cuda'
        cls.dtype = dtype = torch.float16

        sc_cfg = transformers.AutoConfig.from_pretrained("/data/aigc/llm/checkpoints/4l-starcoder/")
        sc_cfg.n_layer = 16
        sc_cfg.n_embd = 1024
        sc_cfg.n_head = 8
        sc_cfg.n_inner = 4096
        sc_cfg.n_positions = 512

        cls.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "/hddlustre/llm/public/checkpoints/pretrained/starcoder/")
        cls.tokenizer.pad_token_id = cls.tokenizer.eos_token_id

        starcoder: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_config(sc_cfg).to(
            dtype=dtype, device=device)
        starcoder.eval()

        cls.model = FlashMQATForCausalLM.from_starcoder(from_model=starcoder, dtype=dtype, device=device)
        cls.model.eval()
        cls.config = cls.model.config

    @torch.no_grad()
    def testGenerate(self):
        seqs = [
            "# This is a print function\ndef",
            "import time\n",
            "assert torch.allclose(logits, sc_logits, atol=5e-3",
            "import torch\n",
        ]
        self.tokenizer.padding_side = "left"
        encoding = self.tokenizer(seqs, return_tensors="pt", padding=True)
        prompt: torch.Tensor = encoding['input_ids'].to(self.device)
        prompt_att_mask: torch.Tensor = encoding['attention_mask'].to(self.device)
        gconfig = GenerationConfig(min_new_tokens=10,
                                   max_new_tokens=100,
                                   temperature=1.0,
                                   greedy=True,
                                   top_k=50,
                                   top_p=1.0,
                                   num_samples=1)

        vg, vglogits, vgmask = vanilla_packed_generate(model=self.model,
                                                       tokenizer=self.tokenizer,
                                                       input_ids=prompt,
                                                       attention_mask=prompt_att_mask,
                                                       gconfig=gconfig)

        g, logits, mask = generate(model=self.model,
                                   tokenizer=self.tokenizer,
                                   input_ids=prompt,
                                   attention_mask=prompt_att_mask,
                                   gconfig=gconfig)

        # print(self.tokenizer.batch_decode(torch.cat([prompt, g], -1)))
        assert torch.allclose(g, vg), (g, vg)
        assert torch.allclose(logits, vglogits,
                              atol=5e-3), (logits, vglogits, (logits - vglogits).abs().max())
        assert torch.allclose(vgmask, mask), (vgmask, mask)


if __name__ == "__main__":
    unittest.main()