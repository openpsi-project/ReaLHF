import itertools
import os
import unittest

import torch
import torch.distributed
import transformers

from reallm.impl.model.nn.flash_mqat.flash_generate import generate, GenerationConfig
from reallm.impl.model.nn.real_llm_api import add_helper_functions, ReaLModel
from reallm.impl.model.nn.real_llm_base import (flash_model_embed_param_count, flash_model_head_param_count,
                                                flash_model_tblock_param_count, FlashMQATBlock, OutputHead,
                                                VocabPositionEmbedding)
from tests.utils import init_global_constants, MODEL_NAME
import reallm.api.huggingface
import reallm.base.constants

torch.cuda.manual_seed_all(2)


class LlamaFlashMQATForwardTest(unittest.TestCase):

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

        cls.hf_path = hf_path = "/lustre/public/pretrained_model_weights/deepseek-coder-6.7b-base"
        # hf_path = "/lustre/public/pretrained_model_weights/Llama-2-13b-hf"
        # hf_path = "/lustre/public/pretrained_model_weights/codellama-13B"

        cls.bs = bs = 3
        cls.device = device = "cuda"
        hf_config = transformers.AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
        hf_config.num_hidden_layers = 2
        hf_config.hidden_size = 256
        hf_config.num_attention_heads = 4
        hf_config.num_key_value_heads = 2
        hf_config.intermediate_size = 1024

        cls.tokenizer = reallm.api.huggingface.load_hf_tokenizer(hf_path)
        cls.tokenizer.pad_token_id = cls.tokenizer.eos_token_id

        cls.llama: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_config(hf_config).to(
            dtype=torch.float16, device=device)
        cls.llama.eval()

        with reallm.base.constants.model_scope(MODEL_NAME):
            cls.hf_like_model = ReaLModel.from_llama(from_model=cls.llama, dtype=torch.float16, device=device)
            cls.hf_like_model = add_helper_functions(cls.hf_like_model)
            cls.hf_like_model.eval()
        cls.config = cls.hf_like_model.config

    def testCountLayerParams(self):

        def count_nn_module_params(m):
            return sum(p.data.numel() for p in m.parameters())

        layers = self.hf_like_model.layers
        for idx, l in enumerate(layers):
            if isinstance(l, VocabPositionEmbedding):
                self.assertEqual(
                    count_nn_module_params(l),
                    flash_model_embed_param_count(self.hf_like_model.config),
                )
            elif isinstance(l, FlashMQATBlock):
                self.assertEqual(
                    count_nn_module_params(l),
                    flash_model_tblock_param_count(self.hf_like_model.config, idx - 1),
                )
            elif isinstance(l, OutputHead):
                self.assertEqual(
                    count_nn_module_params(l),
                    flash_model_head_param_count(self.hf_like_model.config),
                )
            else:
                raise NotImplementedError()

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
        )
        if with_mask:
            x1 = x1 * attention_mask.unsqueeze(-1)
            x2 = x2 * attention_mask.unsqueeze(-1)
        assert torch.allclose(x1, x2, atol=5e-3), ((x1 - x2).abs()).max()

    def testForward(self):
        seqlen_c = [20, 128]
        with_mask_c = [False, True]
        with reallm.base.constants.model_scope(MODEL_NAME):
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

        new_tokens = self.hf_like_model.generate(self.tokenizer, input_ids, attention_mask,
                                                 gconfig=gconfig).sequences

        assert torch.allclose(seq[:, max_prompt_len:], new_tokens), (
            seq,
            torch.cat([input_ids, new_tokens], 1),
            max_prompt_len,
            with_mask,
        )

    def testGenerate(self):
        max_prompt_len_c = [10, 32, 64]
        with_mask_c = [False, True]
        with reallm.base.constants.model_scope(MODEL_NAME):
            for max_prompt_len, with_mask in itertools.product(max_prompt_len_c, with_mask_c):
                self._generate(max_prompt_len, with_mask)

    @unittest.skip("skip because it is slow")
    def testDumpLoad(self):
        os.makedirs("/tmp/_flash_mqat_test/llama/", exist_ok=True)
        ReaLModel.dump_to_llama(
            self.hf_like_model.config,
            self.hf_like_model.state_dict(),
            "/tmp/_flash_mqat_test/llama/",
            self.hf_path,
        )
        from reallm.impl.model.utils.save_load import load_from_disk

        self.llama.load_state_dict(load_from_disk("/tmp/_flash_mqat_test/llama/"))


if __name__ == "__main__":
    # unittest.main(defaultTest="LlamaFlashMQATForwardTest.testDumpLoad")
    unittest.main()
