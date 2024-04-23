import os
import unittest

import torch
import transformers

from reallm.impl.model.nn.flash_mqat.flash_generate import (generate, GenerationConfig, vanilla_cpu_generate,
                                                            vanilla_packed_generate)
from reallm.impl.model.nn.flash_mqat.flash_mqat_base import FlashMQATModel, PipeCacheData, PipeTransferData
from reallm.impl.model.utils.functional import gather_shifted_log_probs
import reallm.api.huggingface

from tests.utils import init_global_constants

try:
    from flash_attn.bert_padding import pad_input, unpad_input
except ModuleNotFoundError:
    pass

torch.random.manual_seed(0)
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


class FlashMQATGPUGPUAccordanceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.bs = bs = 7
        cls.device = device = "cuda"
        cls.dtype = dtype = torch.float16

        sc_cfg = transformers.AutoConfig.from_pretrained("/lustre/meizy/models/starcoder_4l/")
        sc_cfg.n_layer = 16
        sc_cfg.n_embd = 1024
        sc_cfg.n_head = 8
        sc_cfg.n_inner = 4096
        sc_cfg.n_positions = 512

        cls.tokenizer = reallm.api.huggingface.load_hf_tokenizer("/lustre/meizy/models/starcoder_4l/")
        cls.tokenizer.pad_token_id = cls.tokenizer.eos_token_id

        starcoder: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_config(sc_cfg).to(
            dtype=dtype, device=device)
        starcoder.eval()

        cls.model = FlashMQATModel.from_starcoder(from_model=starcoder, dtype=dtype, device=device)
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
        prompt: torch.Tensor = encoding["input_ids"].to(self.device)
        prompt_att_mask: torch.Tensor = encoding["attention_mask"].to(self.device)
        gconfig = GenerationConfig(
            min_new_tokens=10,
            max_new_tokens=100,
            temperature=1.0,
            greedy=True,
            top_k=50,
            top_p=1.0,
            num_samples=1,
        )

        vg, vglogprob, vgmask = vanilla_packed_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=prompt,
            attention_mask=prompt_att_mask,
            gconfig=gconfig,
        )

        g, logprob, mask, _, _ = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=prompt,
            attention_mask=prompt_att_mask,
            gconfig=gconfig,
        )

        # print(self.tokenizer.batch_decode(torch.cat([prompt, g], -1)))
        assert torch.allclose(g, vg), (g, vg)
        assert torch.allclose(logprob, vglogprob, atol=5e-3), (
            logprob,
            vglogprob,
            (logprob - vglogprob).abs().max(),
        )
        assert torch.allclose(mask, vgmask)


class FlashMQATCPUGPUAccordanceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.bs = bs = 7
        cls.device = device = "cpu"
        cls.dtype = dtype = torch.float32

        sc_cfg = transformers.AutoConfig.from_pretrained("/lustre/meizy/models/starcoder_4l/")
        sc_cfg.n_layer = 1
        sc_cfg.n_embd = 1024
        sc_cfg.n_head = 8
        sc_cfg.n_inner = 4096
        sc_cfg.n_positions = 512

        cls.tokenizer = reallm.api.huggingface.load_hf_tokenizer("/lustre/meizy/models/starcoder_4l/")
        cls.tokenizer.pad_token_id = cls.tokenizer.eos_token_id

        starcoder: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_config(sc_cfg).to(
            dtype=dtype, device=device)
        starcoder.eval()

        cls.model = FlashMQATModel.from_starcoder(from_model=starcoder, dtype=dtype, device=device)
        cls.model.eval()
        cls.config = cls.model.config

    @torch.no_grad()
    def testGenerate(self):
        seqs = [
            "# This is a print function\ndef",
            "import time\n",
            "assert torch.allclose(logits, sc_logits, atol=5e-3",
        ]
        self.tokenizer.padding_side = "left"
        encoding = self.tokenizer(seqs, return_tensors="pt", padding=True)
        prompt: torch.Tensor = encoding["input_ids"].to(self.device)
        prompt_att_mask: torch.Tensor = encoding["attention_mask"].to(self.device)
        gconfig = GenerationConfig(
            min_new_tokens=10,
            max_new_tokens=100,
            temperature=1.0,
            greedy=True,
            top_k=50,
            top_p=1.0,
            num_samples=1,
        )
        vcg, vclogprob, vcmask = vanilla_cpu_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=prompt,
            attention_mask=prompt_att_mask,
            gconfig=gconfig,
        )
        self.model = self.model.cuda().to(torch.float16)
        vg, vglogprob, vgmask = vanilla_packed_generate(
            model=self.model.cuda(),
            tokenizer=self.tokenizer,
            input_ids=prompt.cuda(),
            attention_mask=prompt_att_mask.cuda(),
            gconfig=gconfig,
        )
        vglogprob = vglogprob.float().cpu()

        seq = torch.cat([prompt, vcg], -1)
        seq_attn_mask = torch.logical_and(seq.ne(self.tokenizer.pad_token_id),
                                          seq.ne(self.tokenizer.eos_token_id))
        packed_input_ids, pad_indices, cu_seqlens, max_seq_len = unpad_input(seq, seq_attn_mask)
        x = PipeTransferData(cu_seqlens=cu_seqlens.cuda(), max_seqlen=max_seq_len)
        ys = [PipeCacheData(input_ids=packed_input_ids.cuda())
              ] + [PipeCacheData() for _ in range(self.model.config.n_layers + 1)]
        inf_logits = self.model(x, ys).pp_output.float().cpu()
        inf_logits = pad_input(inf_logits, pad_indices, seq.shape[0], seq.shape[1])
        inf_logprob = gather_shifted_log_probs(inf_logits, seq)[:, prompt.shape[1] - 1:]
        assert torch.allclose(vcg, vg.cpu()), (vcg, vg)
        assert torch.allclose(inf_logprob, vclogprob, atol=5e-3), (
            inf_logprob,
            vclogprob,
            (inf_logprob - vclogprob).abs().max(),
        )
        assert torch.allclose(vglogprob, vclogprob, atol=5e-3), (
            vglogprob,
            vclogprob,
            (vglogprob - vclogprob).abs().max(),
        )
        assert torch.allclose(vgmask.cpu(), vcmask)


if __name__ == "__main__":
    unittest.main()
