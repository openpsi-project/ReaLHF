import argparse
import os
import time

# from torch.profiler import profile, ProfilerActivity, record_function
# import transformers
import torch
import torch.multiprocessing as mp

from tests.parallel.utils import *

parser = argparse.ArgumentParser()

parser.add_argument("--num_mp", type=int, default=1)
parser.add_argument("--num_pp", type=int, default=4)
parser.add_argument("--num_dp", type=int, default=1)
parser.add_argument("--num_shards", type=int, default=3)
parser.add_argument("--model_type", type=str, default="llama")
parser.add_argument("--baseline_model_path",
                    type=str,
                    default="/lustre/public/pretrained_model_weights/Llama-2-7b-hf")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--min_new_tokens", type=int, default=10)
parser.add_argument("--max_new_tokens", type=int, default=30)
parser.add_argument("--use_gradient_checkpointing", action="store_true")
parser.add_argument("--use_bf16", action="store_true")
parser.add_argument("--use_sequence_parallel", action="store_true")
parser.add_argument("--test", type=str, default="testInference")

args = parser.parse_args()
setattr(args, "world_size", args.num_mp * args.num_dp * args.num_pp)
if args.model_type == "llama":
    if args.num_pp == 1:
        suffix = f"_{args.num_mp}mp_{args.num_shards}s"
    elif args.num_mp == 1:
        suffix = f"_{args.num_pp}pp_{args.num_shards}s"
    elif args.num_pp > 1:
        suffix = f"_{args.num_pp}pp_{args.num_mp}mp_{args.num_shards}s"
    setattr(args, "model_parallel_path",
            "/lustre/public/pretrained_model_weights/sharded/Llama-2-7b-hf" + suffix)


def init_handles(rank, args):
    device = setup_gpu(rank, args.world_size)
    init_global_constants(args.num_dp, args.num_mp, args.num_pp)
    torch_dist_rank = torch.distributed.get_rank()
    cuda_visible = os.environ["CUDA_VISIBLE_DEVICES"]
    print(f"PROCESS RANK: {rank}; \n"
          f"TORCH DIST RANK: {torch_dist_rank}; \n"
          f"CUDA VISIBLE: {cuda_visible}")

    model = make_model(device, args)
    backend = make_backend(args)
    ft_spec = make_finetune_spec(args.batch_size)
    interface = make_interface(args)
    # ft_spec = None
    # backend = None
    # interface = None

    backend.initialize(model, ft_spec)
    return device, model, backend, interface


def run_inference(rank: int, res_queue: mp.Queue, seed: int):
    device, model, backend, interface = init_handles(rank, args)
    data = init_data(model.tokenizer, device, args.batch_size, seed=seed)

    # packed_input_ids = data['packed_input_ids']
    # cu_seqlens = data['cu_seqlens']
    # max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

    model.module.eval()

    st = time.monotonic()
    # logits = model.module(packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens,
    #                       max_seqlen=max_seqlen).logits.float()
    res = interface.inference(model, data)
    logits = res['logits']
    print(f"rank {rank} mp FIRST inference "
          f"time cost {time.monotonic() - st:.4f}")
    if logits is not None:
        print(f"rank {rank} mp FIRST inference logits shape {logits.shape}")

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #              record_shapes=True,
    #              profile_memory=True,
    #              with_stack=True,
    #              with_flops=True) as prof:
    for _ in range(10):
        st = time.monotonic()
        res = interface.inference(model, data)
        logits = res['logits']
        print(f"rank {rank} mp inference time cost {time.monotonic() - st:.4f}")
    # prof.export_chrome_trace(f"mp{rank}_trace.json")
    # for _ in range(10):
    #     st = time.monotonic()
    #     logits = model.module(packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens,
    #                         max_seqlen=max_seqlen).logits.float()
    #     print(f"rank {rank} mp inference time cost {time.monotonic() - st:.4f}")

    import base.constants
    if base.constants.pipe_parallel_rank() == args.num_pp - 1:
        res_queue.put(logits)
    time.sleep(2)


def run_train_batch(rank: int, res_queue: mp.Queue, seed: int):
    device, model, backend, interface = init_handles(rank, args)
    data = init_data(model.tokenizer, device, args.batch_size, seed=seed)

    st = time.monotonic()
    res = interface.train_step(model, data)
    print(f"rank {rank} mp FIRST train time cost {time.monotonic() - st:.4f}, res {res}")

    for _ in range(10):
        st = time.monotonic()
        res = interface.train_step(model, data)
        print(f"rank {rank} mp train time cost {time.monotonic() - st:.4f}, res {res}")


def run_generate(rank: int, res_queue: mp.Queue, seed: int):
    device, model, backend, interface = init_handles(rank, args)
    data = init_data(model.tokenizer, device, args.batch_size, seed=seed)
    from impl.model.nn.flash_mqat.flash_generate import GenerationConfig
    gconfig = GenerationConfig(min_new_tokens=args.min_new_tokens, max_new_tokens=args.max_new_tokens)

    st = time.monotonic()
    outputs = interface.generate(model, data, gconfig=gconfig)
    t = time.monotonic() - st
    print(f"rank {rank} mp FIRST generate time cost {t:.4f}")
    if len(outputs) > 0:
        print(f"generate result gen_tokens shape{outputs['gen_tokens'].shape}, "
              f"log probs shape {outputs['log_probs'].shape}")

    for i in range(10):
        data = init_data(model.tokenizer, device, args.batch_size, seed=seed)
        st = time.monotonic()
        outputs = interface.generate(model, data, gconfig=gconfig)
        t = time.monotonic() - st
        print(f"rank {rank} mp generate time cost {t:.4f}")
        if len(outputs) > 0:
            print(f"generate result gen_tokens shape{outputs['gen_tokens'].shape}, "
                  f"log probs shape {outputs['log_probs'].shape}")


def run_linear(rank: int, res_queue: mp.Queue, seed: int):
    import torch.distributed as dist

    from impl.model.utils.model_parallel.modules import ColumnParallelLinear, RowParallelLinear
    import base.constants

    torch.cuda.set_device(rank)
    device = torch.cuda.current_device()
    print(device)
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:12345",
        world_size=args.world_size,
        rank=rank,
    )
    import deepspeed
    deepspeed.init_distributed()
    init_global_constants(args.num_dp, args.num_mp, 1)

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


class ModelParallelFlashMQATTest:

    @classmethod
    def setUpClass(cls):
        clear_name_resolve()
        cls.baseline_model = None

    def init_tokenizer(self):
        import api.huggingface

        self.tokenizer = api.huggingface.load_hf_tokenizer(args.baseline_model_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    def init_baseline_model(self):
        from impl.model.nn.flash_mqat.flash_mqat_api import HuggingfaceLikeFlashMQATForCausalLM
        import impl.model.nn.flash_mqat.flash_from_hf_impl
        self.device = device = 'cuda'
        self.dtype = dtype = torch.float16

        self.init_tokenizer()

        if args.model_type == "llama":
            self.baseline_model = HuggingfaceLikeFlashMQATForCausalLM.from_llama(
                model_path=args.baseline_model_path,)
        self.baseline_model.to(dtype=dtype, device=device)

    def testTrainStep(self):
        clear_name_resolve()
        self.seed = random.randint(1, 10000)
        self.res_queue = mp.Queue(maxsize=128)
        setup_barrier(args.world_size)
        self.pipe_model_processes = [
            mp.Process(target=run_train_batch, args=(i, self.res_queue, self.seed))
            for i in range(args.world_size)
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
        setup_barrier(args.world_size)
        self.pipe_model_processes = [
            mp.Process(target=run_inference, args=(i, self.res_queue, self.seed))
            for i in range(args.world_size)
        ]
        for p in self.pipe_model_processes:
            p.start()

        res = []
        for _ in range(args.num_mp):
            res.append(self.res_queue.get())

        for p in self.pipe_model_processes:
            p.join()

        print(f"res[0] shape {res[0].shape}")

    @torch.no_grad()
    def testInferenceAccordance(self):
        clear_name_resolve()
        self.seed = 1
        self.res_queue = mp.Queue(maxsize=128)
        setup_barrier(args.world_size)
        self.pipe_model_processes = [
            mp.Process(target=run_inference, args=(i, self.res_queue, self.seed))
            for i in range(args.world_size)
        ]
        for p in self.pipe_model_processes:
            p.start()

        res = []
        for _ in range(args.num_mp):
            res.append(self.res_queue.get())

        for p in self.pipe_model_processes:
            p.join()

        self.init_baseline_model()
        data = init_data(self.tokenizer, self.device, args.batch_size, seed=self.seed, dp_rank=0, num_dp=1)
        packed_input_ids = data['packed_input_ids']
        cu_seqlens = data['cu_seqlens']
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        self.baseline_model.eval()

        st = time.monotonic()
        r = self.baseline_model(packed_input_ids=packed_input_ids,
                                cu_seqlens=cu_seqlens,
                                max_seqlen=max_seqlen).logits.float()
        print(f"baseline FIRST inference time cost {time.monotonic() - st:.4f}")

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #              record_shapes=True,
        #              profile_memory=True,
        #              with_stack=True,
        #              with_flops=True) as prof:
        for _ in range(10):
            st = time.monotonic()
            r = self.baseline_model(packed_input_ids=packed_input_ids,
                                    cu_seqlens=cu_seqlens,
                                    max_seqlen=max_seqlen).logits.float()
            print(f"baseline inference time cost {time.monotonic() - st:.4f}")
        # prof.export_chrome_trace("baseline_trace.json")

        print(f"diff: {r - res[0]}, max/correct_max {(r - res[0]).abs().max()}/{r.abs().max()}, "
              f" mean {(r - res[0]).abs().mean()},")

        # import base.consistency
        # base.consistency.check_all_model_parallel(NUM_MP)

    @torch.no_grad()
    def testLinearAccordance(self):
        self.seed = random.randint(1, 10000)
        self.res_queue = mp.Queue(maxsize=128)
        setup_barrier(args.world_size)
        self.pipe_model_processes = [
            mp.Process(target=run_linear, args=(i, self.res_queue, self.seed)) for i in range(args.world_size)
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
        setup_barrier(args.world_size)
        self.pipe_model_processes = [
            mp.Process(target=run_generate, args=(i, self.res_queue, self.seed))
            for i in range(args.world_size)
        ]
        for p in self.pipe_model_processes:
            p.start()
        for p in self.pipe_model_processes:
            p.join()


if __name__ == "__main__":
    test = ModelParallelFlashMQATTest()
    test.setUpClass()

    test_func = getattr(test, args.test)
    test_func()
