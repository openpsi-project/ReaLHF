import unittest

import pytest
import torch
import transformers

from impl.model.nn.flash_mqat import FlashMQATForCausalLM, PipeCacheData, PipeTransferData
import api.huggingface


class LlamaFlashMQATForwardTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.bs = bs = 4
        cls.device = device = "cuda"
        hf_config = transformers.AutoConfig.from_pretrained("/lustre/fw/pretrained/llama-7b")
        hf_config.num_hidden_layers = 2
        hf_config.hidden_size = 256
        hf_config.num_attention_heads = 4
        hf_config.num_key_value_heads = 2
        hf_config.intermediate_size = 1024

        cls.tokenizer = api.huggingface.load_hf_tokenizer("/lustre/fw/pretrained/llama-7b")
        cls.tokenizer.pad_token_id = cls.tokenizer.eos_token_id

        cls.llama: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_config(hf_config).to(
            dtype=torch.float16, device=device)
        cls.llama.eval()

        cls.model = FlashMQATForCausalLM.from_llama(from_model=cls.llama, dtype=torch.float16, device=device)
        cls.model.eval()
        cls.config = cls.model.config

    @torch.no_grad()
    def testMain(self):
        seqlen = 20
        input_ids = torch.randint(0, self.config.vocab_size, (self.bs, seqlen)).to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        x1 = self.llama(input_ids=input_ids, attention_mask=attention_mask).logits.half()
        x = PipeTransferData()
        ys = [PipeCacheData(input_ids=input_ids)] + [PipeCacheData() for _ in range(self.config.n_layers + 1)]
        x2 = self.model(x, ys).pp_output
        assert torch.allclose(x1, x2, atol=5e-3), ((x1 - x2).abs()).max()


if __name__ == "__main__":
    unittest.main()
