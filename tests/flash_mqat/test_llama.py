import itertools
import unittest

import pytest
import torch
import transformers

from impl.model.nn.flash_mqat import (FlashMQATForCausalLM, generate, GenerationConfig,
                                      HuggingfaceLikeFlashMQATForCausalLM, PipeCacheData, PipeTransferData)
import api.huggingface


class LlamaFlashMQATForwardTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.bs = bs = 3
        cls.device = device = "cuda"
        hf_config = transformers.AutoConfig.from_pretrained(
            "/lustre/public/pretrained_model_weights/Llama-2-13b-hf")
        hf_config.num_hidden_layers = 2
        hf_config.hidden_size = 256
        hf_config.num_attention_heads = 4
        hf_config.num_key_value_heads = 2
        hf_config.intermediate_size = 1024

        cls.tokenizer = api.huggingface.load_hf_tokenizer(
            "/lustre/public/pretrained_model_weights/Llama-2-13b-hf")
        cls.tokenizer.pad_token_id = cls.tokenizer.eos_token_id

        cls.llama: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_config(hf_config).to(
            dtype=torch.float16, device=device)
        cls.llama.eval()

        cls.model = FlashMQATForCausalLM.from_llama(from_model=cls.llama, dtype=torch.float16, device=device)
        cls.model.eval()
        cls.hf_like_model = HuggingfaceLikeFlashMQATForCausalLM(cls.model)
        cls.config = cls.model.config

    @torch.no_grad()
    def _hf_like_forward(self, with_mask: bool, seqlen: int):
        input_ids = torch.randint(0, self.config.vocab_size, (self.bs, seqlen)).to(self.device)
        if with_mask:
            input_lens = torch.randint(1, seqlen, (self.bs,), dtype=torch.long).to(self.device)
            attention_mask = torch.arange(seqlen, device=self.device).unsqueeze(0) < input_lens.unsqueeze(1)
        else:
            attention_mask = None
        x1 = self.llama(input_ids=input_ids, attention_mask=attention_mask).logits.half()
        x2 = self.hf_like_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits
        if with_mask:
            x1 = x1 * attention_mask.unsqueeze(-1)
            x2 = x2 * attention_mask.unsqueeze(-1)
        assert torch.allclose(x1, x2, atol=5e-3), ((x1 - x2).abs()).max()

    def testForward(self):
        seqlen_c = [20, 128]
        with_mask_c = [False, True]
        for with_mask, seqlen in itertools.product(with_mask_c, seqlen_c):
            self._hf_like_forward(with_mask, seqlen)

    @torch.no_grad()
    def _generate(self, max_prompt_len: int, with_mask: bool):
        input_ids = torch.randint(0, self.config.vocab_size, (self.bs, max_prompt_len)).to(self.device)
        input_lens = torch.randint(1, max_prompt_len, (self.bs,), dtype=torch.long).to(self.device)
        if with_mask:
            attention_mask = torch.arange(max_prompt_len - 1, -1, -1,
                                          device=self.device).unsqueeze(0) < input_lens.unsqueeze(1)
        else:
            attention_mask = None

        gconfig = GenerationConfig(min_new_tokens=1, max_new_tokens=10, greedy=True)
        seq = self.llama.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            min_new_tokens=gconfig.min_new_tokens,
            max_new_tokens=gconfig.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        new_tokens, *_ = generate(self.model, self.tokenizer, input_ids, attention_mask, gconfig=gconfig)

        assert torch.allclose(seq[:, max_prompt_len:], new_tokens), (
            seq,
            torch.cat([input_ids, new_tokens], 1),
            max_prompt_len,
            with_mask,
        )

    def testGenerate(self):
        max_prompt_len_c = [10, 32, 64]
        with_mask_c = [False, True]
        for max_prompt_len, with_mask in itertools.product(max_prompt_len_c, with_mask_c):
            self._generate(max_prompt_len, with_mask)


if __name__ == "__main__":
    unittest.main(defaultTest="LlamaFlashMQATForwardTest.testForward")
