import json
import queue
import time
import unittest

import torch
import transformers
import viztracer

import impl.model.nn.flash_mqat as flash_mqat


class InflightBatchingThroughputTest(unittest.TestCase):

    def setUp(self):
        self.bs = 4
        self.n_prompts = 16
        assert self.n_prompts % self.bs == 0

        self.device = "cuda"
        model_path = "/lustre/fw/pretrained/gpt2"
        self.model = flash_mqat.FlashMQATForCausalLM.from_gpt2(model_path=model_path,
                                                               dtype=torch.float16,
                                                               device=self.device)
        self.model.eval()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        with open("/lustre/fw/datasets/imdb/rl/ppo_prompt.jsonl", "r") as f:
            self.prompts_str = [json.loads(line)["prompt"] for line in f][:self.n_prompts]

        self.max_prompt_len = 10

        self.tokenizer.padding_side = "left"
        self.pad_prompt_encodings = self.tokenizer(
            self.prompts_str,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_len,
            return_tensors="pt",
        )
        for k, v in self.pad_prompt_encodings.items():
            self.pad_prompt_encodings[k] = v.to(self.device)

        self.nonpad_prompt_encodings = self.tokenizer(
            self.prompts_str,
            padding=False,
            truncation=True,
            max_length=self.max_prompt_len,
        )

        self.gconfig = flash_mqat.GenerationConfig(
            min_new_tokens=10,
            max_new_tokens=512,
            greedy=False,
            top_k=50,
        )

    def _main_defaultgen(self):
        tik = time.perf_counter()
        answers = []
        for i in range(self.n_prompts // self.bs):
            s = slice(i * self.bs, (i + 1) * self.bs)
            gen_tokens, *_ = flash_mqat.generate(
                self.model,
                self.tokenizer,
                input_ids=self.pad_prompt_encodings["input_ids"][s],
                attention_mask=self.pad_prompt_encodings["attention_mask"][s],
                gconfig=self.gconfig,
            )
            answers.append(gen_tokens)
        t1 = time.perf_counter() - tik
        n_tokens = 0
        for a in answers:
            seqlens = (
                (a != self.tokenizer.eos_token_id).logical_and(a != self.tokenizer.pad_token_id).sum(1) +
                1).clip(max=self.gconfig.max_new_tokens)
            n_tokens += seqlens.sum()
        print(f"Default generate {n_tokens} in {t1:.3f}s, Throughput: {n_tokens/t1} tokens/s")

    def _main_inflightgen(self):
        factor = 2
        inqueue = queue.Queue(maxsize=self.n_prompts * 2)
        outqueue = queue.Queue(maxsize=self.n_prompts * 2)
        for _ in range(factor):
            for i in range(self.n_prompts):
                inqueue.put(
                    torch.tensor(
                        self.nonpad_prompt_encodings["input_ids"][i],
                        device=self.device,
                        dtype=torch.long,
                    ))
        generator = flash_mqat.InflightBatchingGenerator(
            inqueue,
            outqueue,
            self.model,
            self.tokenizer,
            self.gconfig,
            self.bs,
            self.max_prompt_len,
        )

        tik = time.perf_counter()
        all_res = []
        cnt_ = 0
        while len(all_res) < self.n_prompts:
            while True:
                try:
                    all_res.append(outqueue.get_nowait())
                except queue.Empty:
                    break
            generator.step_for(1)
            cnt_ += generator.batch_size
        t2 = time.perf_counter() - tik

        print(f"Inflight generate {cnt_} tokens in {t2:.3f} s, Throughput: {cnt_ / t2} tokens/s")

    def testMain(self):
        tracer = viztracer.VizTracer(tracer_entries=int(1e6), max_stack_depth=10)
        tracer.start()
        self._main_defaultgen()
        self._main_inflightgen()
        tracer.stop()
        tracer.save("result.json")


if __name__ == "__main__":
    unittest.main()
