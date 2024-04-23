import os
import unittest

try:
    from flash_attn.bert_padding import pad_input, unpad_input
except ModuleNotFoundError:
    pass

import torch
import transformers

from reallm.impl.model.nn.flash_mqat.flash_mqat_base import FlashMQATModel, PipeCacheData, PipeTransferData
import reallm.api.huggingface

from tests.utils import init_global_constants


class FlashMQATGPT2Test(unittest.TestCase):

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

        model_path = "/lustre/public/pretrained_model_weights/testOnly/gpt2-4l"

        cls.tokenizer = reallm.api.huggingface.load_hf_tokenizer(model_path)
        cls.tokenizer.pad_token_id = cls.tokenizer.eos_token_id

        cls.gpt: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
            model_path).to(dtype=dtype, device=device)
        cls.gpt.eval()

        cls.model = FlashMQATModel.from_gpt2(model_path=model_path, dtype=dtype, device=device)
        cls.model.eval()
        cls.config = cls.model.config

    @torch.no_grad()
    def testStandardForward(self):
        config = self.config
        bs = self.bs
        gpt = self.gpt
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
        sc_logits = gpt(input_ids=input_ids).logits
        logits = model(x, ys).pp_output
        assert torch.allclose(logits, sc_logits, atol=2e-5), ((logits - sc_logits)).abs().max()

        # right pad
        x = PipeTransferData(attention_mask=rightpad_attention_mask)
        ys = [PipeCacheData(input_ids=input_ids)] + [PipeCacheData() for _ in range(config.n_layers + 1)]
        sc_logits = gpt(input_ids=input_ids,
                        attention_mask=rightpad_attention_mask).logits * rightpad_attention_mask.unsqueeze(-1)
        logits = model(x, ys).pp_output * rightpad_attention_mask.unsqueeze(-1)
        assert torch.allclose(logits, sc_logits, atol=2e-5), ((logits - sc_logits)).abs().max()

        # left pad
        x = PipeTransferData(attention_mask=leftpad_attention_mask)
        ys = [PipeCacheData(input_ids=input_ids)] + [PipeCacheData() for _ in range(config.n_layers + 1)]
        sc_logits = gpt(input_ids=input_ids,
                        attention_mask=leftpad_attention_mask).logits * leftpad_attention_mask.unsqueeze(-1)
        logits = model(x, ys).pp_output * leftpad_attention_mask.unsqueeze(-1)
        assert torch.allclose(logits, sc_logits, atol=2e-5), ((logits - sc_logits)).abs().max()


if __name__ == "__main__":
    unittest.main()
