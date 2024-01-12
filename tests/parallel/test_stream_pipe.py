import argparse
import os
import time
import unittest

# import transformers
import torch
import torch.multiprocessing as mp

from base.monitor import get_tracer
from tests.utils import *
import api.config as config_package

NUM_MP = 1
NUM_PP = 4
NUM_DP = 2
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
BATCH_SIZE = 128
MIN_NEW_TOKENS = 10
MAX_NEW_TOKENS = 30

USE_GRADIENT_CHECKPOINTING = True
USE_BF16 = False
USE_SEQ_PARALLEL = True
GRADIENT_ACCUMULATION_FUSION = False


def make_model(device):
    import api.model
    import impl.model.nn.flash_mqat.flash_mqat_api
    model_config = config_package.Model("flash_mqat_actor",
                                        args=dict(
                                            model_path=MODEL_PARALLEL_PATH,
                                            from_type=MODEL_TYPE,
                                            tokenizer_path=MODEL_PARALLEL_PATH,
                                            init_from_scratch=True,
                                            no_param_instantiation=True,
                                            dtype="bf16" if USE_BF16 else "fp16",
                                        ))
    assert NUM_PP > 1, "can not test stream pipe without pipeline parallel"
    if NUM_MP == 1:
        model_config.wrappers += [
            config_package.ModelWrapper("pipe",
                                        args=dict(
                                            model_path=MODEL_PARALLEL_PATH,
                                            num_pp=NUM_PP,
                                            num_dp=NUM_DP,
                                            is_critic=False,
                                            init_critic_from_actor=False,
                                            init_from_scratch=False,
                                            partition_method="parameters_balanced",
                                        ))
        ]
    else:
        model_config.wrappers += [
            config_package.ModelWrapper("model_pipe_parallel",
                                        args=dict(
                                            model_path=MODEL_PARALLEL_PATH,
                                            num_pp=NUM_PP,
                                            num_mp=NUM_MP,
                                            num_dp=NUM_DP,
                                            sequence_parallel=USE_SEQ_PARALLEL,
                                            gradient_accumulation_fusion=GRADIENT_ACCUMULATION_FUSION,
                                            is_critic=False,
                                            init_critic_from_actor=False,
                                            init_from_scratch=False,
                                            partition_method="parameters_balanced",
                                        ))
        ]

    model = api.model.make_model(model_config, name=MODEL_NAME, device=device)
    return model


def make_stream_pipe_backend():
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
                                              engine_type="stream_pipe",
                                              gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
                                              num_pipeline_stages=NUM_PP,
                                              enable_fp16=not USE_BF16,
                                              enable_bf16=USE_BF16,
                                              sequence_parallel=USE_SEQ_PARALLEL)))


def make_interface():
    import api.model
    return api.model.make_interface(config_package.ModelInterface(type_="flash_sft", args=dict()))


def make_stream_pipe_interface():
    import api.model
    return api.model.make_interface(config_package.ModelInterface(type_="stream_pipe_test", args=dict()))


def init_handles(rank):
    device = setup_gpu(rank, WORLD_SIZE)
    init_global_constants(NUM_DP, NUM_MP, NUM_PP)
    torch_dist_rank = torch.distributed.get_rank()
    cuda_visible = os.environ["CUDA_VISIBLE_DEVICES"]
    print(f"PROCESS RANK: {rank}; \n"
          f"TORCH DIST RANK: {torch_dist_rank}; \n"
          f"CUDA VISIBLE: {cuda_visible}")

    model = make_model(device)
    backend = make_stream_pipe_backend()
    ft_spec = make_finetune_spec(BATCH_SIZE)
    interface = make_stream_pipe_interface()

    backend.initialize(model, ft_spec)
    return device, model, backend, interface


def run_train_batch(rank, seed):
    device, model, backend, interface = init_handles(rank)
    engine = model.module
    from impl.model.backend.pipe_engine.stream_pipe_engine import StreamPipeEngine
    assert isinstance(engine, StreamPipeEngine)
    data = init_data(model.tokenizer, device, BATCH_SIZE, seed=seed)
    engine.enable_async_p2p()

    # os.environ["DLLM_TRACE"] = "1"
    tracer = get_tracer(tracer_entries=int(2e6),
                        max_stack_depth=10,
                        ignore_c_function=False,
                        ignore_frozen=True,
                        log_async=True,
                        min_duration=10,
                        output_file=f"/home/meizy/logs/viztracer/trace{rank}.json")
    tracer.start()

    st = time.monotonic()
    future, data = interface.train_step(model, data, num_micro_batches=2 * NUM_PP)

    while not future.done():
        engine.run()

    res = interface.postprocess_train_step(model, data, future)
    print(f"rank {rank} mp FIRST train time cost {time.monotonic() - st:.4f}, res {res}")

    for _ in range(3):
        st = time.monotonic()
        future, data = interface.train_step(model, data, num_micro_batches=2 * NUM_PP)

        while not future.done():
            engine.run()

        res = interface.postprocess_train_step(model, data, future)
        print(f"rank {rank} mp train time cost {time.monotonic() - st:.4f}, res {res}")

    time.sleep(1)
    engine.stop_controller()
    tracer.save()


def run_generate(rank, seed):
    device, model, backend, interface = init_handles(rank)
    engine = model.module
    from impl.model.backend.pipe_engine.stream_pipe_engine import StreamPipeEngine
    assert isinstance(engine, StreamPipeEngine)

    data = init_data(model.tokenizer, device, 2 * BATCH_SIZE, seed=seed)
    from impl.model.nn.flash_mqat.flash_generate import GenerationConfig
    gconfig = GenerationConfig(min_new_tokens=MIN_NEW_TOKENS, max_new_tokens=MAX_NEW_TOKENS)
    # engine.enable_async_p2p()

    # os.environ["DLLM_TRACE"] = "1"
    tracer = get_tracer(
        tracer_entries=int(2e6),
        # max_stack_depth=10,
        ignore_c_function=False,
        ignore_frozen=True,
        log_async=True,
        min_duration=20,
        output_file=f"/home/meizy/logs/viztracer/trace{rank}.json")
    tracer.start()

    st = time.monotonic()
    future, data = interface.generate(model, data, gconfig=gconfig, num_micro_batches=4 * NUM_PP)

    while not future.done():
        engine.run()

    res = interface.postprocess_generate(model, data, future)

    print(f"rank {rank} mp FIRST generate time cost {time.monotonic() - st:.4f}, batchsize {2*BATCH_SIZE}")

    # for _ in range(3):
    #     st = time.monotonic()
    #     future, data = interface.generate(model, data, gconfig=gconfig, num_micro_batches=4*NUM_PP)

    #     while not future.done():
    #         engine.run()

    #     res = interface.postprocess_generate(model, data, future)
    #     print(f"rank {rank} mp generate time cost {time.monotonic() - st:.4f}")

    # if len(res) > 0:
    #     print(f"generate result gen_tokens shape{res['gen_tokens'].shape}, "
    #           f"log probs shape {res['log_probs'].shape}")

    engine.stop_controller()
    tracer.save()


def run_mixed(rank, seed):
    device, model, backend, interface = init_handles(rank)
    engine = model.module
    from impl.model.backend.pipe_engine.stream_pipe_engine import StreamPipeEngine
    from impl.model.interface.pipe.stream_pipe_test_interface import StreamPipeTestInterface
    assert isinstance(engine, StreamPipeEngine)
    assert isinstance(interface, StreamPipeTestInterface)
    # os.environ["DLLM_TRACE"] = "1"
    tracer = get_tracer(tracer_entries=int(2e6),
                        max_stack_depth=10,
                        ignore_c_function=False,
                        ignore_frozen=True,
                        log_async=True,
                        min_duration=20,
                        output_file=f"/home/meizy/logs/viztracer/async_p2p/trace{rank}.json")
    tracer.start()

    train_iters = 3
    train_datas = [init_data(model.tokenizer, device, BATCH_SIZE, seed=seed + i) for i in range(train_iters)]
    gen_data = init_data(model.tokenizer, device, BATCH_SIZE * 2, seed=seed + 100)

    from impl.model.nn.flash_mqat.flash_generate import GenerationConfig
    gconfig = GenerationConfig(min_new_tokens=MIN_NEW_TOKENS, max_new_tokens=MAX_NEW_TOKENS)
    engine.enable_async_p2p()

    st = time.monotonic()
    gf, _ = interface.generate(model, gen_data, gconfig=gconfig, num_micro_batches=4 * NUM_PP)
    # for train_data in train_datas:
    tfs = []
    for i in range(train_iters):
        tf, _ = interface.train_step(model, train_datas[i], num_micro_batches=2 * NUM_PP)
        tfs.append(tf)

        while not tf.done():
            engine.run()

        print(f"rank {rank} train step {i} done, time {time.monotonic() - st}")

    while not gf.done():
        engine.run()

    gres = interface.postprocess_generate(model, gen_data, gf)
    tress = []
    for tf, train_data in zip(tfs, train_datas):
        tres = interface.postprocess_train_step(model, train_data, tf)
        tress.append(tres)
    print(f"first mixed time cost {time.monotonic() - st:.4f}")

    if len(gres) > 0:
        print(f"generate result gen_tokens shape{gres['gen_tokens'].shape}, "
              f"log probs shape {gres['log_probs'].shape}")
    print(f"tres {tress}")

    engine.stop_controller()
    tracer.save()


class StreamPipeTest(unittest.TestCase):

    def testGenerate(self):
        clear_name_resolve()
        self.seed = 1
        setup_barrier(WORLD_SIZE)
        self.pipe_model_processes = [
            mp.Process(target=run_generate, args=(i, self.seed)) for i in range(WORLD_SIZE)
        ]
        for p in self.pipe_model_processes:
            p.start()

        for p in self.pipe_model_processes:
            p.join()

    def testTrainStep(self):
        clear_name_resolve()
        self.seed = 1
        setup_barrier(WORLD_SIZE)
        self.pipe_model_processes = [
            mp.Process(target=run_train_batch, args=(i, self.seed)) for i in range(WORLD_SIZE)
        ]
        for p in self.pipe_model_processes:
            p.start()

        for p in self.pipe_model_processes:
            p.join()

    def testMixed(self):
        clear_name_resolve()
        self.seed = 1
        setup_barrier(WORLD_SIZE)
        self.pipe_model_processes = [
            mp.Process(target=run_mixed, args=(i, self.seed)) for i in range(WORLD_SIZE)
        ]
        for p in self.pipe_model_processes:
            p.start()

        for p in self.pipe_model_processes:
            p.join()


if __name__ == "__main__":
    unittest.main(defaultTest="StreamPipeTest.testMixed")