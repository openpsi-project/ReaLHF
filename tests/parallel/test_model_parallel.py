import functools
import os
import time
import unittest

from torch.profiler import profile, ProfilerActivity, record_function
# import transformers
import torch
import torch.distributed
import torch.multiprocessing as mp

from tests.utils import *
import api.config as config_package

# TODO: organize parallel testing codes, merge pipe_parallel_test.py and model_parallel_test.py

NUM_MP = 2
NUM_PP = 4
NUM_DP = 1
NUM_SHARDS = 3
WORLD_SIZE = NUM_MP * NUM_DP * NUM_PP
MODEL_TYPE = "llama"
if MODEL_TYPE == "llama":
    if NUM_PP == 1:
        SUFFIX = f"_{NUM_MP}mp_{NUM_SHARDS}s"
    elif NUM_MP == 1:
        SUFFIX = f"_{NUM_PP}pp_{NUM_SHARDS}s"
    elif NUM_PP > 1:
        SUFFIX = f"_{NUM_PP}pp_{NUM_MP}mp_{NUM_SHARDS}s"
    # BASELINE_MODEL_PATH = "/home/meizy/models/test/Llama-2-4l"
    # MODEL_PARALLEL_PATH = f"/lustre/public/pretrained_model_weights/sharded/Llama-2-4l{SUFFIX}"
    BASELINE_MODEL_PATH = "/lustre/public/pretrained_model_weights/Llama-2-7b-hf"
    MODEL_PARALLEL_PATH = f"/lustre/public/pretrained_model_weights/sharded/Llama-2-7b-hf{SUFFIX}"
BATCH_SIZE = 512
MIN_NEW_TOKENS = 10
MAX_NEW_TOKENS = 30

USE_GRADIENT_CHECKPOINTING = True
USE_BF16 = False
USE_SEQ_PARALLEL = True
GRADIENT_ACCUMULATION_FUSION = False


def make_backend():
    import api.model

    if NUM_PP == 1:
        return api.model.make_backend(
            config_package.ModelBackend(
                type_="ds_train",
                args=dict(
                    optimizer_name="adam",
                    optimizer_config=dict(lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95)),
                    warmup_steps_proportion=0.0,
                    min_lr_ratio=0.0,
                    # TODO: test zero_stage = 2 or 3 later
                    gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
                    zero_stage=1,
                    enable_fp16=not USE_BF16,
                    enable_bf16=USE_BF16,
                ),
            ))
    elif NUM_PP > 1:
        return api.model.make_backend(
            config_package.ModelBackend(
                type_="ds_train",
                args=dict(
                    optimizer_name="adam",
                    optimizer_config=dict(lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95)),
                    warmup_steps_proportion=0.0,
                    min_lr_ratio=0.0,
                    zero_stage=1,
                    engine_type="pipe",
                    gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
                    num_pipeline_stages=NUM_PP,
                    enable_fp16=not USE_BF16,
                    enable_bf16=USE_BF16,
                    sequence_parallel=USE_SEQ_PARALLEL,
                    num_pipeline_micro_batches=NUM_PP,
                ),
            ))


def make_interface():
    import api.model

    return api.model.make_interface(config_package.ModelInterface(type_="flash_sft", args=dict()))


def make_model(device):
    import api.model
    import impl.model.nn.flash_mqat.flash_mqat_api

    model_config = config_package.Model(
        "flash_mqat",
        args=dict(
            model_path=MODEL_PARALLEL_PATH,
            from_type="self" if NUM_PP == 1 else "empty_actor",
            dtype="bf16" if USE_BF16 else "fp16",
            hf_model_type=MODEL_TYPE,
            tokenizer_path=MODEL_PARALLEL_PATH,
            sequence_parallel=USE_SEQ_PARALLEL,
            gradient_accumulation_fusion=False,
        ),
    )
    assert NUM_PP > 1 or NUM_MP > 1, "can not test model without mp or dp"
    if NUM_PP > 1:
        model_config.wrappers += [
            config_package.ModelWrapper(
                "pipe_flash_mqat",
                args=dict(
                    model_path=MODEL_PARALLEL_PATH,
                    partition_method="parameters_balanced",
                    init_critic_from_actor=False,
                    init_from_scratch=False,
                ),
            )
        ]

    model = api.model.make_model(model_config, name=MODEL_NAME, device=device)
    return model


def init_handles(rank):
    device = setup_gpu(rank, WORLD_SIZE)
    init_global_constants(NUM_DP, NUM_MP, NUM_PP)
    torch_dist_rank = torch.distributed.get_rank()
    cuda_visible = os.environ["CUDA_VISIBLE_DEVICES"]
    print(f"PROCESS RANK: {rank}; \n"
          f"TORCH DIST RANK: {torch_dist_rank}; \n"
          f"CUDA VISIBLE: {cuda_visible}")

    model = make_model(device)
    backend = make_backend()
    ft_spec = make_finetune_spec(BATCH_SIZE)
    interface = make_interface()
    # ft_spec = None
    # backend = None
    # interface = None

    backend.initialize(model, ft_spec)
    return device, model, backend, interface


def run_inference(rank: int, res_queue: mp.Queue, seed: int):
    device, model, backend, interface = init_handles(rank)
    data = init_data(model.tokenizer, device, BATCH_SIZE, seed=seed)

    # packed_input_ids = data['packed_input_ids']
    # cu_seqlens = data['cu_seqlens']
    # max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

    model.module.eval()

    st = time.monotonic()
    res = interface.inference(model, data)
    logits = res["logits"]
    print(f"rank {rank} mp FIRST inference "
          f"time cost {time.monotonic() - st:.4f}")
    if logits is not None:
        print(f"rank {rank} mp FIRST inference logits shape {logits.shape}")

    st = time.monotonic()
    res = interface.inference(model, data)
    logits = res["logits"]
    if logits is not None:
        from impl.model.parallelism.model_parallel.mappings import gather_from_tensor_model_parallel_region

        logits = gather_from_tensor_model_parallel_region(logits)
        print(f"rank {rank} mp inference time cost {time.monotonic() - st:.4f}")

    import base.constants

    if base.constants.pipe_parallel_rank() == NUM_PP - 1:
        res_queue.put(logits)
    time.sleep(2)


def run_train_batch(rank: int, res_queue: mp.Queue, seed: int):
    device, model, backend, interface = init_handles(rank)
    data = init_data(model.tokenizer, device, BATCH_SIZE, seed=seed)

    st = time.monotonic()
    res = interface.train_step(model, data)
    print(f"rank {rank} mp FIRST train time cost {time.monotonic() - st:.4f}, res {res}")

    # for _ in range(3):
    #     st = time.monotonic()
    #     res = interface.train_step(model, data)
    #     print(f"rank {rank} mp train time cost {time.monotonic() - st:.4f}, res {res}")


def run_generate(rank: int, res_queue: mp.Queue, seed: int):
    device, model, backend, interface = init_handles(rank)
    data = init_data(model.tokenizer, device, BATCH_SIZE, seed=seed)
    from impl.model.nn.flash_mqat.flash_generate import GenerationConfig

    gconfig = GenerationConfig(min_new_tokens=MIN_NEW_TOKENS, max_new_tokens=MAX_NEW_TOKENS)

    st = time.monotonic()
    outputs = interface.generate(model, data, gconfig=gconfig)
    t = time.monotonic() - st
    print(f"rank {rank} mp FIRST generate time cost {t:.4f}")
    if len(outputs) > 0:
        print(f"generate result gen_tokens shape{outputs['gen_tokens'].shape}, "
              f"log probs shape {outputs['log_probs'].shape}")

    # for i in range(10):
    #     data = init_data(model.tokenizer, device, BATCH_SIZE, seed=seed)
    #     st = time.monotonic()
    #     outputs = interface.generate(model, data, gconfig=gconfig)
    #     t = time.monotonic() - st
    #     print(f"rank {rank} mp generate time cost {t:.4f}")
    #     if len(outputs) > 0:
    #         print(f"generate result gen_tokens shape{outputs['gen_tokens'].shape}, "
    #               f"log probs shape {outputs['log_probs'].shape}")


def run_linear(rank: int, res_queue: mp.Queue, seed: int):
    import torch.distributed as dist

    from impl.model.parallelism.model_parallel.modules import ColumnParallelLinear, RowParallelLinear
    import base.constants

    # device, model, backend, interface = init_handles(rank)

    torch.cuda.set_device(rank)
    device = torch.cuda.current_device()
    print(device)
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:12345",
        world_size=WORLD_SIZE,
        rank=rank,
    )
    import deepspeed

    deepspeed.init_distributed()
    init_global_constants(NUM_DP, NUM_MP, 1)

    torch.manual_seed(seed)
    input = torch.randn(32, 1024, dtype=torch.float, device=device) * 0.02
    w1 = torch.randn(2048, 1024, dtype=torch.float, device=device) * 0.02
    w2 = torch.randn(1024, 2048, dtype=torch.float, device=device) * 0.02
    print(input)
    print(w1)
    print(w2)

    col = ColumnParallelLinear(1024, 2048, bias=False, dtype=torch.float, device=device)
    row = RowParallelLinear(2048, 1024, bias=False, dtype=torch.float, device=device)

    mp_rank = base.constants.model_parallel_rank()
    mp_ws = base.constants.model_parallel_world_size()
    w1 = w1.split(2048 // mp_ws, dim=0)[mp_rank]
    w2 = w2.split(2048 // mp_ws, dim=1)[mp_rank]
    col.load_state_dict({"weight": w1})
    row.load_state_dict({"weight": w2})

    a = col(input)
    b = row(a)

    print(f"rank {rank} linear output {b}")
    if rank == 0:
        res_queue.put(b)
    time.sleep(1)


class ModelParallelFlashMQATTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        clear_name_resolve()
        cls.baseline_model = None

    def init_tokenizer(self):
        import api.huggingface

        self.tokenizer = api.huggingface.load_hf_tokenizer(BASELINE_MODEL_PATH)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    def init_baseline_model(self):
        from impl.model.nn.flash_mqat.flash_mqat_api import forward_helper, generate_helper
        from impl.model.nn.flash_mqat.flash_mqat_base import FlashMQATModel
        import impl.model.nn.flash_mqat.flash_from_hf_impl

        self.device = device = "cuda"
        self.dtype = dtype = torch.float16

        self.init_tokenizer()

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
        init_global_constants(1, 1, 1, model_name="baseline")
        self.baseline_model = getattr(FlashMQATModel, f"from_{MODEL_TYPE}")(model_path=BASELINE_MODEL_PATH,
                                                                            dtype=dtype,
                                                                            device=device)
        self.baseline_model.forward = functools.partial(forward_helper, self.baseline_model)
        self.baseline_model.generate = functools.partial(generate_helper, self.baseline_model)

    def testTrainStep(self):
        clear_name_resolve()
        self.seed = random.randint(1, 10000)
        self.res_queue = mp.Queue(maxsize=128)
        setup_barrier(WORLD_SIZE)
        self.pipe_model_processes = [
            mp.Process(target=run_train_batch, args=(i, self.res_queue, self.seed)) for i in range(WORLD_SIZE)
        ]
        for p in self.pipe_model_processes:
            p.start()

        for p in self.pipe_model_processes:
            p.join()

    @torch.no_grad()
    def testInference(self):
        clear_name_resolve()
        self.seed = random.randint(1, 10000)
        self.res_queue = mp.Queue(maxsize=128)
        setup_barrier(WORLD_SIZE)
        self.pipe_model_processes = [
            mp.Process(target=run_inference, args=(i, self.res_queue, self.seed)) for i in range(WORLD_SIZE)
        ]
        for p in self.pipe_model_processes:
            p.start()

        res = []
        for _ in range(NUM_MP):
            res.append(self.res_queue.get())

        for p in self.pipe_model_processes:
            p.join()

        print(f"res[0] shape {res[0].shape}")

    @torch.no_grad()
    def testInferenceAccordance(self):
        clear_name_resolve()
        self.seed = 1
        self.res_queue = mp.Queue(maxsize=128)
        setup_barrier(WORLD_SIZE)
        self.pipe_model_processes = [
            mp.Process(target=run_inference, args=(i, self.res_queue, self.seed)) for i in range(WORLD_SIZE)
        ]
        for p in self.pipe_model_processes:
            p.start()

        res = []
        for _ in range(NUM_MP):
            res.append(self.res_queue.get())

        for p in self.pipe_model_processes:
            p.join()

        self.init_baseline_model()
        data = init_data(self.tokenizer, self.device, BATCH_SIZE, seed=self.seed, dp_rank=0, num_dp=1)
        packed_input_ids = data["packed_input_ids"]
        cu_seqlens = data["cu_seqlens"].int()
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
        self.baseline_model.eval()

        st = time.monotonic()
        r = self.baseline_model(packed_input_ids=packed_input_ids,
                                cu_seqlens=cu_seqlens,
                                max_seqlen=max_seqlen).float()
        print(f"baseline FIRST inference time cost {time.monotonic() - st:.4f}")

        st = time.monotonic()
        r = self.baseline_model(packed_input_ids=packed_input_ids,
                                cu_seqlens=cu_seqlens,
                                max_seqlen=max_seqlen).float()
        print(f"baseline inference time cost {time.monotonic() - st:.4f}")

        print(f"diff: {r - res[0]}, max/correct_max {(r - res[0]).abs().max()}/{r.abs().max()}, "
              f" mean {(r - res[0]).abs().mean()},")

        # import base.consistency
        # base.consistency.check_all_model_parallel(NUM_MP)

    @torch.no_grad()
    def testLinearAccordance(self):
        self.seed = random.randint(1, 10000)
        self.res_queue = mp.Queue(maxsize=128)
        setup_barrier(WORLD_SIZE)
        self.pipe_model_processes = [
            mp.Process(target=run_linear, args=(i, self.res_queue, self.seed)) for i in range(WORLD_SIZE)
        ]
        for p in self.pipe_model_processes:
            p.start()
        res = []
        # for _ in range(WORLD_SIZE):
        res.append(self.res_queue.get())
        for p in self.pipe_model_processes:
            p.join()

        torch.manual_seed(self.seed)
        input = torch.randn(32, 1024, dtype=torch.float, device=torch.cuda.current_device()) * 0.02
        w1 = torch.randn(2048, 1024, dtype=torch.float, device=torch.cuda.current_device()) * 0.02
        w2 = torch.randn(1024, 2048, dtype=torch.float, device=torch.cuda.current_device()) * 0.02
        print(input)
        print(w1)
        print(w2)

        col = torch.nn.Linear(1024, 2048, bias=False, dtype=torch.float, device=torch.cuda.current_device())
        row = torch.nn.Linear(2048, 1024, bias=False, dtype=torch.float, device=torch.cuda.current_device())
        col.load_state_dict({"weight": w1})
        row.load_state_dict({"weight": w2})
        r = row(col(input))
        print(f"normal linear output {r}")
        print(f"diff: {r - res[0]} {(r - res[0]).abs().max()}")

    def testGenerate(self):
        clear_name_resolve()
        self.seed = 1
        self.res_queue = mp.Queue(maxsize=128)
        setup_barrier(WORLD_SIZE)
        self.pipe_model_processes = [
            mp.Process(target=run_generate, args=(i, self.res_queue, self.seed)) for i in range(WORLD_SIZE)
        ]
        for p in self.pipe_model_processes:
            p.start()
        for p in self.pipe_model_processes:
            p.join()


if __name__ == "__main__":
    unittest.main(defaultTest="ModelParallelFlashMQATTest.testTrainStep")
