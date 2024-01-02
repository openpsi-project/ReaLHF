import os
import random

import torch
import torch.multiprocessing as mp

import api.config as config_package
import base.gpu_utils
import base.name_resolve as name_resolve
import base.names as names

EXPR_NAME = "test"
TRIAL_NAME = "test"
MODEL_NAME = "test_model"
WORKER_TYPE = "model_worker"

BARRIER = None


def setup_barrier(world_size):
    global BARRIER
    BARRIER = mp.Barrier(world_size)


def setup_gpu(rank, world_size):
    os.environ["DLLM_MODE"] = "LOCAL"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    BARRIER.wait()
    base.gpu_utils.isolate_cuda_device(WORKER_TYPE, rank, world_size, EXPR_NAME, TRIAL_NAME)
    print(f"rank {rank} isolated cuda device")
    BARRIER.wait()
    base.gpu_utils.reveal_ddp_identity_single_model(EXPR_NAME, TRIAL_NAME, MODEL_NAME, rank)
    print(f"rank {rank} revealed ddp identity")
    BARRIER.wait()
    base.gpu_utils.setup_ddp_single_model(EXPR_NAME, TRIAL_NAME, MODEL_NAME, rank)
    device = torch.device('cuda', 0)
    print(f"rank {rank} setup ddp")
    import deepspeed
    deepspeed.init_distributed()
    print(f"rank {rank} setup deepspeed")
    return device


def clear_name_resolve():
    name_resolve.clear_subtree(names.trial_root(experiment_name=EXPR_NAME, trial_name=TRIAL_NAME))


def make_finetune_spec(bs_per_device, total_train_epochs=1, total_train_steps=10, steps_per_epoch=10):
    import api.model
    finetune_spec = api.model.FinetuneSpec(
        total_train_epochs=total_train_epochs,
        total_train_steps=total_train_steps,
        steps_per_epoch=steps_per_epoch,
        batch_size_per_device=bs_per_device,
    )
    return finetune_spec


def random_sentence(min_len=50, max_len=100):
    words = ["the", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog"]
    sentence_length = random.randint(min_len, max_len)
    return " ".join(random.choices(words, k=sentence_length))
    # return "Output less than 50 words:"


def make_input(tokenizer, device, s):
    tokenizer.padding_side = "left"
    prompts = tokenizer(s, return_tensors="pt", padding=True)

    input_ids, attention_mask = prompts["input_ids"], prompts["attention_mask"]
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    return input_ids, attention_mask


def make_batch(tokenizer, device, batch_size, dp_rank, dp_worldsize, seed=373):
    random.seed(seed)
    whole_batch = [random_sentence() for _ in range(batch_size)]
    dp_batch = whole_batch[batch_size // dp_worldsize * dp_rank:batch_size // dp_worldsize * (dp_rank + 1)]
    return make_input(tokenizer, device, dp_batch)


def init_global_constants(num_dp, num_mp, num_pp):
    from base.topology import PipelineParallelGrid, PipeModelDataParallelTopology
    topo = PipeModelDataParallelTopology(num_dp=num_dp, num_mp=num_mp, num_pp=num_pp)
    ws = num_dp * num_mp * num_pp
    import torch.distributed as dist
    wg = dist.new_group(ranks=list(range(ws)))
    grid = PipelineParallelGrid(topology=topo, process_group=wg)
    import base.constants
    base.constants.set_experiment_trial_names(EXPR_NAME, TRIAL_NAME)
    base.constants.set_parallelism_group(MODEL_NAME, wg)
    base.constants.set_model_name(MODEL_NAME)
    base.constants.set_grid(MODEL_NAME, grid)


def init_data(tokenizer, device, batch_size, seed, dp_rank=None, num_dp=None):
    from flash_attn.bert_padding import unpad_input

    import base.constants
    import base.namedarray
    if dp_rank == None:
        assert num_dp == None
        dp_rank = base.constants.data_parallel_rank()
        num_dp = base.constants.data_parallel_world_size()
    input_ids, attention_mask = make_batch(tokenizer, device, batch_size, dp_rank % num_dp, num_dp, seed=seed)
    packed_input_ids, _, cu_seqlens, max_seqlen = unpad_input(input_ids, attention_mask)
    prompt_mask = torch.zeros_like(packed_input_ids).bool()
    data = base.namedarray.NamedArray(
        packed_input_ids=packed_input_ids,
        cu_seqlens=cu_seqlens,
        prompts=input_ids,
        prompt_mask=prompt_mask,
        prompt_att_mask=attention_mask,
    )
    return data


def make_backend(args):
    import api.model
    if args.num_pp == 1:
        return api.model.make_backend(
            config_package.ModelBackend(
                type_='ds_train',
                args=dict(
                    optimizer_name='adam',
                    optimizer_config=dict(lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95)),
                    warmup_steps_proportion=0.0,
                    min_lr_ratio=0.0,
                    # TODO: test zero_stage = 2 or 3 later
                    gradient_checkpointing=args.use_gradient_checkpointing,
                    zero_stage=1,
                    enable_fp16=not args.use_bf16,
                    enable_bf16=args.use_bf16,
                )))
    elif args.num_pp > 1:
        return api.model.make_backend(
            config_package.ModelBackend(type_='ds_train',
                                        args=dict(
                                            optimizer_name='adam',
                                            optimizer_config=dict(lr=1e-5,
                                                                  weight_decay=0.0,
                                                                  betas=(0.9, 0.95)),
                                            warmup_steps_proportion=0.0,
                                            min_lr_ratio=0.0,
                                            zero_stage=1,
                                            engine_type="pipe",
                                            gradient_checkpointing=args.use_gradient_checkpointing,
                                            num_pipeline_stages=args.num_pp,
                                            enable_fp16=not args.use_bf16,
                                            enable_bf16=args.use_bf16,
                                            sequence_parallel=args.use_sequence_parallel,
                                        )))


def make_stream_pipe_backend(args):
    import api.model
    return api.model.make_backend(
        config_package.ModelBackend(type_='ds_train',
                                    args=dict(
                                        optimizer_name='adam',
                                        optimizer_config=dict(lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95)),
                                        warmup_steps_proportion=0.0,
                                        min_lr_ratio=0.0,
                                        zero_stage=1,
                                        engine_type="stream_pipe",
                                        gradient_checkpointing=args.use_gradient_checkpointing,
                                        num_pipeline_stages=args.num_pp,
                                        enable_fp16=not args.use_bf16,
                                        enable_bf16=args.use_bf16,
                                        sequence_parallel=args.use_sequence_parallel,
                                    )))


def make_interface(args):
    import api.model
    return api.model.make_interface(config_package.ModelInterface(type_="flash_sft", args=dict()))


def make_stream_pipe_interface(args):
    import api.model
    return api.model.make_interface(config_package.ModelInterface(type_="stream_pipe_test", args=dict()))


def make_model(device, args):
    import api.model
    import impl.model.nn.flash_mqat.flash_mqat_api
    model_config = config_package.Model("flash_mqat_actor",
                                        args=dict(
                                            model_path=args.model_parallel_path,
                                            from_type=args.model_type,
                                            tokenizer_path=args.model_parallel_path,
                                            init_from_scratch=True,
                                            no_param_instantiation=True,
                                            dtype="bf16" if args.use_bf16 else "fp16",
                                        ))
    assert args.num_pp > 1 or args.num_mp > 1, "can not test model without mp or dp"
    if args.num_pp == 1:
        model_config.wrappers += [
            config_package.ModelWrapper("model_parallel",
                                        args=dict(
                                            model_path=args.model_parallel_path,
                                            sequence_parallel=args.use_sequence_parallel,
                                            is_critic=False,
                                            init_critic_from_actor=False,
                                            init_from_scratch=False,
                                        ))
        ]
    elif args.num_mp == 1:
        model_config.wrappers += [
            config_package.ModelWrapper("pipe",
                                        args=dict(
                                            model_path=args.model_parallel_path,
                                            num_pp=args.num_pp,
                                            num_dp=args.num_dp,
                                            is_critic=False,
                                            init_critic_from_actor=False,
                                            init_from_scratch=False,
                                        ))
        ]
    elif args.num_pp > 1:
        model_config.wrappers += [
            config_package.ModelWrapper("model_pipe_parallel",
                                        args=dict(
                                            model_path=args.model_parallel_path,
                                            num_pp=args.num_pp,
                                            num_mp=args.num_mp,
                                            num_dp=args.num_dp,
                                            sequence_parallel=args.use_sequence_parallel,
                                            is_critic=False,
                                            init_critic_from_actor=False,
                                            init_from_scratch=False,
                                        ))
        ]

    model = api.model.make_model(model_config, name=MODEL_NAME, device=device)
    return model
