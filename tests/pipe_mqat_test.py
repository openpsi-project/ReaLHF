import logging
import os
import random
import time
import unittest

import torch
import torch.multiprocessing as mp

from base.namedarray import NamedArray
import api.config as config_package
import base.gpu_utils
import base.name_resolve as name_resolve
import base.names as names

# import transformers

# mp.set_start_method('spawn', force=True) # this will make global barrier not work

EXPR_NAME = "test"
TRIAL_NAME = "test"
MODEL_NAME = "pipedatamodel"
MODEL_TYPE = "model_worker"
NUM_PP = 4
NUM_DP = 1
WORLD_SIZE = NUM_PP * NUM_DP
BASELINE_MODEL_PATH = "/lustre/meizy/models/starcoder_4l"
PIPELINE_MODEL_PATH = F"/lustre/meizy/models/pipe_starcoder_4l_{NUM_PP}pp_{NUM_DP}s"
BATCH_SIZE = 8
MIN_NEW_TOKENS = 20
MAX_NEW_TOKENS = 20

BARRIER = mp.Barrier(WORLD_SIZE)
LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"
# for plotting
logging.basicConfig(filename="/home/meizy/logs/pipe_mqat.log",
                    filemode="w",
                    format=LOG_FORMAT,
                    datefmt=DATE_FORMAT,
                    level="DEBUG")
# logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level="DEBUG")

logger = logging.getLogger("pipe_mqat_test")


def setup_gpu(rank):
    os.environ["DLLM_MODE"] = "LOCAL"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    BARRIER.wait()
    base.gpu_utils.isolate_cuda_device(MODEL_TYPE, rank, WORLD_SIZE, EXPR_NAME, TRIAL_NAME)
    BARRIER.wait()
    base.gpu_utils.reveal_ddp_identity(EXPR_NAME, TRIAL_NAME, MODEL_NAME, rank)
    BARRIER.wait()
    world_size, ddp_rank, local_gpu_id = base.gpu_utils.setup_ddp(EXPR_NAME, TRIAL_NAME, MODEL_NAME, rank)
    device = torch.device('cuda', 0)
    import deepspeed
    deepspeed.init_distributed()
    return device


def clear_name_resolve():
    name_resolve.clear_subtree(names.trial_root(experiment_name=EXPR_NAME, trial_name=TRIAL_NAME))


def make_pipe_interface():
    import api.model
    return api.model.make_interface(config_package.ModelInterface(type_="pipe_flash_sft", args=dict()))


def make_finetune_spec(bs_per_device):
    import api.model
    finetune_spec = api.model.FinetuneSpec(
        total_train_epochs=1,
        total_train_steps=10,
        steps_per_epoch=10,
        batch_size_per_device=bs_per_device,
    )
    return finetune_spec


def make_pipe_backend():
    import api.model
    return api.model.make_backend(
        config_package.ModelBackend(type_='ds_train',
                                    args=dict(optimizer_name='adam',
                                              optimizer_config=dict(lr=1e-5,
                                                                    weight_decay=0.0,
                                                                    betas=(0.9, 0.95)),
                                              warmup_steps_proportion=0.0,
                                              min_lr_ratio=0.0,
                                              zero_stage=1,
                                              engine_type="pipe",
                                              num_pipeline_stages=NUM_PP)))


def make_pipe_model(model_path, device):
    from impl.model.backend.ds_pipe_engine import PipeDataParallelTopology
    import api.model
    topology = PipeDataParallelTopology(num_pp=NUM_PP, num_dp=NUM_DP)
    model_config = config_package.Model(type_="starcoder_flash_mqat_pipe",
                                        args=dict(
                                            model_path=model_path,
                                            num_pp=NUM_PP,
                                            num_dp=NUM_DP,
                                            load_from_full_ckpt=False,
                                            dtype=torch.float16,
                                        ))
    model = api.model.make_model(model_config, name=MODEL_NAME, device=device)
    return topology, model


def make_input(tokenizer, device, s):
    tokenizer.padding_side = "left"
    prompts = tokenizer(s, return_tensors="pt", padding=True)

    input_ids, attention_mask = prompts["input_ids"], prompts["attention_mask"]
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    return input_ids, attention_mask


def random_sentence(min_len=1, max_len=10):
    words = ["the", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog"]
    sentence_length = random.randint(min_len, max_len)
    return " ".join(random.choices(words, k=sentence_length))


def make_batch(tokenizer, device, batch_size, dp_rank, dp_worldsize, seed=373):
    random.seed(seed)
    whole_batch = [random_sentence() for _ in range(batch_size)]
    dp_batch = whole_batch[batch_size // dp_worldsize * dp_rank:batch_size // dp_worldsize * (dp_rank + 1)]
    return make_input(tokenizer, device, dp_batch)


def init_handles(rank):
    device = setup_gpu(rank)

    rank = torch.distributed.get_rank()
    cuda_visible = os.environ["CUDA_VISIBLE_DEVICES"]
    logger.info(f"PROCESS RANK: {rank}; \n"
                f"TORCH DIST RANK: {torch.distributed.get_rank()}; \n"
                f"CUDA VISIBLE: {cuda_visible}")
    topo, model = make_pipe_model(PIPELINE_MODEL_PATH, device)
    backend = make_pipe_backend()
    ft_spec = make_finetune_spec(BATCH_SIZE)
    interface = make_pipe_interface()

    backend.initialize(model, ft_spec)

    return device, model, backend, interface


def init_data(rank, model, device, seed):
    from impl.model.utils.data import build_packed_inputs
    input_ids, attention_mask = make_batch(model.tokenizer,
                                           device,
                                           BATCH_SIZE,
                                           rank % NUM_DP,
                                           NUM_DP,
                                           seed=seed)
    packed_input_ids, cu_seqlens, max_seqlen = build_packed_inputs(input_ids, attention_mask)
    prompt_mask = torch.zeros_like(packed_input_ids)
    data = NamedArray(
        packed_input_ids=packed_input_ids,
        cu_seqlens=cu_seqlens,
        prompt_mask=prompt_mask,
    )
    return data


def pipe_generate(rank, res_queue: mp.Queue, seed: int):
    device, model, backend, interface = init_handles(rank)
    data = init_data(rank, model, device, seed)

    from impl.model.nn.flash_mqat import GenerationConfig
    gconfig = GenerationConfig(min_new_tokens=MIN_NEW_TOKENS, max_new_tokens=MAX_NEW_TOKENS)
    st = time.monotonic()
    outputs = interface.generate(model, data, gconfig=gconfig)
    t = time.monotonic() - st
    # logger.info(input_ids)
    if len(outputs) > 0 and res_queue is not None:
        # logger.info(input_ids)
        # logger.info(outputs["gen_tokens"])
        # logger.info(outputs["log_probs"])
        res_queue.put(outputs["gen_tokens"])
        res_queue.put(outputs["log_probs"])
        res_queue.put(t)
        time.sleep(1)  # wait for queue get, otherwise invalid tensor handle


def pipe_train_batch(rank, res_queue: mp.Queue, seed: int):
    device, model, backend, interface = init_handles(rank)
    data = init_data(rank, model, device, seed)
    outputs = interface.train_step(model, data)
    logger.info(f"{rank} {outputs}")


class PipeFlashMQATTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # name_resolve.clear_subtree(names.trainer_ddp_peer(EXPR_NAME, TRIAL_NAME, MODEL_NAME))
        clear_name_resolve()
        cls.baseline_model = None

    def init_baseline_model(self):
        import transformers

        from impl.model.nn.flash_mqat import FlashMQATForCausalLM, generate, GenerationConfig
        import api.huggingface

        self.device = device = 'cuda'
        self.dtype = dtype = torch.float16

        self.tokenizer = api.huggingface.load_hf_tokenizer("/lustre/meizy/models/starcoder_4l")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        starcoder: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path="/lustre/meizy/models/starcoder_4l").to(dtype=dtype, device=device)
        starcoder.eval()

        self.baseline_model = FlashMQATForCausalLM.from_starcoder(from_model=starcoder,
                                                                  dtype=dtype,
                                                                  device=device)
        self.baseline_model.eval()

    @torch.no_grad()
    def testGenerateAccordance(self):
        clear_name_resolve()
        self.seed = random.randint(0, 1000)
        self.res_queue = mp.Queue(maxsize=128)
        self.pipe_model_processes = [
            mp.Process(target=pipe_generate, args=(i, self.res_queue, self.seed)) for i in range(WORLD_SIZE)
        ]
        for p in self.pipe_model_processes:
            p.start()

        g = self.res_queue.get()
        logprob = self.res_queue.get()
        t = self.res_queue.get()

        for p in self.pipe_model_processes:
            p.join()

        from impl.model.nn.flash_mqat import generate, GenerationConfig
        self.gconfig = GenerationConfig(min_new_tokens=MIN_NEW_TOKENS, max_new_tokens=MAX_NEW_TOKENS)
        # baseline model calculate
        self.init_baseline_model()
        prompt, prompt_att_mask = make_batch(self.tokenizer, self.device, BATCH_SIZE, 0, 1, seed=self.seed)

        s2 = time.monotonic()
        vg, vlogprob, vmask, _ = generate(model=self.baseline_model,
                                          tokenizer=self.tokenizer,
                                          input_ids=prompt,
                                          attention_mask=prompt_att_mask,
                                          gconfig=self.gconfig)
        t2 = time.monotonic() - s2
        print("pipe time:", t)
        print("vanilla time:", t2)

        assert torch.allclose(g, vg), (g, vg)
        assert torch.allclose(logprob, vlogprob, atol=0, rtol=0.01), (logprob, vlogprob)

    @torch.no_grad()
    def testGenerate(self):
        clear_name_resolve()
        self.seed = 1
        self.res_queue = mp.Queue(maxsize=128)
        self.pipe_model_processes = [
            mp.Process(target=pipe_generate, args=(i, self.res_queue, self.seed)) for i in range(WORLD_SIZE)
        ]
        for p in self.pipe_model_processes:
            p.start()

        g = self.res_queue.get()
        logprob = self.res_queue.get()
        t = self.res_queue.get()

        for p in self.pipe_model_processes:
            p.join()

        print(g)
        print(logprob)
        print(t)

    def testTrainBatch(self):
        clear_name_resolve()

        self.seed = random.randint(0, 1000)
        self.res_queue = mp.Queue(maxsize=128)
        self.pipe_model_processes = [
            mp.Process(target=pipe_train_batch, args=(i, self.res_queue, self.seed))
            for i in range(WORLD_SIZE)
        ]
        for p in self.pipe_model_processes:
            p.start()
        for p in self.pipe_model_processes:
            p.join()


if __name__ == "__main__":
    unittest.main(defaultTest="PipeFlashMQATTest.testGenerateAccordance")
