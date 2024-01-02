import argparse
import os
import time
import unittest

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
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--min_new_tokens", type=int, default=10)
parser.add_argument("--max_new_tokens", type=int, default=30)
parser.add_argument("--use_gradient_checkpointing", action="store_true")
parser.add_argument("--use_bf16", action="store_true")
parser.add_argument("--use_sequence_parallel", action="store_true")
parser.add_argument("--test", type=str, default="testTrainStep")

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
    backend = make_stream_pipe_backend(args)
    ft_spec = make_finetune_spec(args.batch_size)
    interface = make_stream_pipe_interface(args)

    backend.initialize(model, ft_spec)
    return device, model, backend, interface


def run_train_batch(rank, seed):
    device, model, backend, interface = init_handles(rank, args)
    engine = model.module
    from impl.model.backend.pipe_engine.stream_pipe_engine import StreamPipeEngine
    assert isinstance(engine, StreamPipeEngine)
    data = init_data(model.tokenizer, device, args.batch_size, seed=seed)

    st = time.monotonic()
    future, data = interface.train_step(model, data)

    while not future.done():
        engine.run()

    res = interface.postprocess_train_step(model, data, future)
    print(f"rank {rank} mp FIRST train time cost {time.monotonic() - st:.4f}, res {res}")

    for _ in range(10):
        st = time.monotonic()
        future, data = interface.train_step(model, data)

        while not future.done():
            engine.run()

        res = interface.postprocess_train_step(model, data, future)
        print(f"rank {rank} mp train time cost {time.monotonic() - st:.4f}, res {res}")

    time.sleep(1)
    engine.stop_controller()


def run_generate(rank, seed):
    device, model, backend, interface = init_handles(rank, args)
    engine = model.module
    from impl.model.backend.pipe_engine.stream_pipe_engine import StreamPipeEngine
    assert isinstance(engine, StreamPipeEngine)

    data = init_data(model.tokenizer, device, args.batch_size, seed=seed)
    from impl.model.nn.flash_mqat.flash_generate import GenerationConfig
    gconfig = GenerationConfig(min_new_tokens=args.min_new_tokens, max_new_tokens=args.max_new_tokens)

    st = time.monotonic()
    future, data = interface.generate(model, data, gconfig=gconfig)

    while not future.done():
        engine.run()

    res = interface.postprocess_generate(model, data, future)

    print(f"rank {rank} mp FIRST generate time cost {time.monotonic() - st:.4f}")

    for _ in range(10):
        st = time.monotonic()
        future, data = interface.generate(model, data, gconfig=gconfig)

        while not future.done():
            engine.run()

        res = interface.postprocess_generate(model, data, future)
        print(f"rank {rank} mp generate time cost {time.monotonic() - st:.4f}, res {res}")

    if len(res) > 0:
        print(f"generate result gen_tokens shape{res['gen_tokens'].shape}, "
              f"log probs shape {res['log_probs'].shape}")

    engine.stop_controller()


def run_mixed(rank, seed):
    device, model, backend, interface = init_handles(rank, args)
    engine = model.module
    from impl.model.backend.pipe_engine.stream_pipe_engine import StreamPipeEngine
    assert isinstance(engine, StreamPipeEngine)

    train_data = init_data(model.tokenizer, device, args.batch_size, seed=seed)
    gen_data = init_data(model.tokenizer, device, args.batch_size, seed=seed + 100)

    from impl.model.nn.flash_mqat.flash_generate import GenerationConfig
    gconfig = GenerationConfig(min_new_tokens=args.min_new_tokens, max_new_tokens=args.max_new_tokens)

    st = time.monotonic()
    gf, _ = interface.generate(model, gen_data, gconfig=gconfig)
    tf, _ = interface.train_step(model, train_data)

    while not gf.done() or not tf.done():
        engine.run()

    gres = interface.postprocess_generate(model, gen_data, gf)
    tres = interface.postprocess_train_step(model, train_data, tf)

    if len(gres) > 0:
        print(f"generate result gen_tokens shape{gres['gen_tokens'].shape}, "
              f"log probs shape {gres['log_probs'].shape}")
    print(f"tres {tres}")
    print(f"mixed time cost {time.monotonic() - st:.4f}")

    engine.stop_controller()


class StreamPipeTest:

    def testGenerate(self):
        clear_name_resolve()
        self.seed = 1
        setup_barrier(args.world_size)
        self.pipe_model_processes = [
            mp.Process(target=run_generate, args=(i, self.seed)) for i in range(args.world_size)
        ]
        for p in self.pipe_model_processes:
            p.start()

        for p in self.pipe_model_processes:
            p.join()

    def testTrainStep(self):
        clear_name_resolve()
        self.seed = 1
        setup_barrier(args.world_size)
        self.pipe_model_processes = [
            mp.Process(target=run_train_batch, args=(i, self.seed)) for i in range(args.world_size)
        ]
        for p in self.pipe_model_processes:
            p.start()

        for p in self.pipe_model_processes:
            p.join()

    def testMixed(self):
        clear_name_resolve()
        self.seed = 1
        setup_barrier(args.world_size)
        self.pipe_model_processes = [
            mp.Process(target=run_mixed, args=(i, self.seed)) for i in range(args.world_size)
        ]
        for p in self.pipe_model_processes:
            p.start()

        for p in self.pipe_model_processes:
            p.join()


if __name__ == "__main__":
    test = StreamPipeTest()

    test_func = getattr(test, args.test)
    test_func()
