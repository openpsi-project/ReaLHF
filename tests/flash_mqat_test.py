import unittest
import torch
import transformers

from impl.model.nn.flash_mqat import FlashMQATForCausalLM, PipeCacheData, PipeTransferData
from impl.model.utils.flash_generate import generate, GenerationConfig, vanilla_packed_generate, vanilla_cpu_generate


class FlashMQATStarCoderTest(unittest.TestCase):

    # @classmethod
    # def setUpClass(cls):
    #     cls.bs = bs = 4
    #     cls.device = device = 'cuda'

    #     sc_cfg = transformers.AutoConfig.from_pretrained("/data/aigc/llm/checkpoints/4l-starcoder/")
    #     sc_cfg.n_layer = 1
    #     sc_cfg.n_embd = 1024
    #     sc_cfg.n_head = 8
    #     sc_cfg.n_inner = 4096
    #     sc_cfg.vocab_size = 5000
    #     sc_cfg.n_positions = 512

    #     cls.tokenizer = transformers.AutoTokenizer.from_pretrained(
    #         "/hddlustre/llm/public/checkpoints/pretrained/starcoder/")
    #     cls.tokenizer.pad_token_id = cls.tokenizer.eos_token_id

    #     cls.starcoder: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_config(
    #         sc_cfg).to(dtype=torch.float16, device=device)
    #     cls.starcoder.eval()

    #     # cls.starcoder = transformers.AutoModelForCausalLM.from_pretrained(
    #     #     "/data/aigc/llm/checkpoints/4l-starcoder/", torch_dtype=torch.float16, use_cache=True).cuda()
    #     # cls.starcoder.eval()

    #     # cls.model = FlashMQATForCausalLM.from_starcoder(model_path="/data/aigc/llm/checkpoints/4l-starcoder/",
    #     #                                                 dtype=torch.float16,
    #     #                                                 device=device)
    #     cls.model = FlashMQATForCausalLM.from_starcoder(from_model=cls.starcoder,
    #                                                     dtype=torch.float16,
    #                                                     device=device)
    #     cls.model.eval()
    #     cls.config = cls.model.config

    @unittest.skip('')
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
        assert torch.allclose(logits, sc_logits), ((logits - sc_logits)).abs().max()

    @unittest.skip('')
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

    @unittest.skip('')
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
        # generated, glogits, glmask = generate(model=self.model,
        #                                       tokenizer=self.tokenizer,
        #                                       input_ids=input_ids,
        #                                       attention_mask=attention_mask,
        #                                       gconfig=gconfig)
        vg, vglogits, vglmask = vanilla_packed_generate(model=self.model,
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
            )
            tam = torch.logical_and(tseq.not_equal(self.tokenizer.pad_token_id),
                                    (tseq.not_equal(self.tokenizer.eos_token_id))).long()
            tlogits = self.starcoder(input_ids=tseq, attention_mask=tam).logits
            tlogits = tlogits.gather(-1, tseq[..., None]).squeeze(-1)
            tseq = tseq[:, input_ids.shape[1]:]
            tlogits = tlogits[:, input_ids.shape[1]:]
        print(vg, tseq)
        print(vglogits, tlogits)
        assert torch.allclose(vglogits, tlogits), (vglogits - tlogits).abs().max()
        # assert torch.allclose(generated, vg)
        # assert torch.allclose(glogits, vglogits), (glogits, vglogits, (glogits - vglogits).abs().max())
        # assert torch.allclose(glmask, vglmask)
        # print(">>>>>>>>>>>>>>>>>>>>")
        # print(input_ids)
        # print(generated)
        # print(self.tokenizer.batch_decode(torch.cat([input_ids, generated], -1)))
        # print("<<<<<<<<<<<<<<<<<<<")
        # print(vg)
        # print(self.tokenizer.batch_decode(torch.cat([input_ids, vg], -1)))


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

    @unittest.skip('')
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
        vglogits = vglogits.gather(-1, vg_input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)[:, input_ids.shape[1] - 1:]
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


if __name__ == "__main__":
    unittest.main()