from typing import *

from reallm.api.core.config import ModelBackend
from reallm.api.core.dfg import ModelInterfaceType, ModelRPC, OffloadHook, SyncParamHook
from reallm.api.core.system_api import ModelName
from reallm.api.quickstart.device_mesh import DeviceMesh, RPCAllocation
from reallm.api.quickstart.model import get_real_model_config, ModelTrainEvalConfig, ParallelismConfig
from reallm.base.topology import PipeModelDataParallelTopology


def get_topo(
    parallel: ParallelismConfig,
    gradient_checkpointing: bool,
    max_prompt_len: Optional[int] = None,
) -> PipeModelDataParallelTopology:
    return PipeModelDataParallelTopology(
        num_mp=parallel.model_parallel_size,
        num_pp=parallel.pipeline_parallel_size,
        num_dp=parallel.data_parallel_size,
        sequence_parallel=parallel.use_sequence_parallel,
        gradient_checkpointing=gradient_checkpointing,
        max_prompt_len=max_prompt_len,
    )


def get_world_size(parallel: ParallelismConfig) -> int:
    return (parallel.model_parallel_size * parallel.pipeline_parallel_size * parallel.data_parallel_size)


def make_train_backend_config(model_cfg: ModelTrainEvalConfig, parallel_cfg: ParallelismConfig):
    if parallel_cfg.pipeline_parallel_size > 1:
        engine_type = "pipe"
    else:
        engine_type = "deepspeed"
    return ModelBackend(
        "ds_train",
        args=dict(
            optimizer_name="adam",
            optimizer_config=dict(
                lr=model_cfg.optimizer.lr,
                weight_decay=model_cfg.optimizer.weight_decay,
                eps=model_cfg.optimizer.eps,
                betas=(model_cfg.optimizer.beta1, model_cfg.optimizer.beta2),
            ),
            lr_scheduler_type=model_cfg.optimizer.lr_scheduler_type,
            warmup_steps_proportion=model_cfg.optimizer.warmup_steps_proportion,
            min_lr_ratio=model_cfg.optimizer.min_lr_ratio,
            zero_stage=(model_cfg.zero_stage if parallel_cfg.pipeline_parallel_size == 1 else min(
                model_cfg.zero_stage, 1)),
            engine_type=engine_type,
            offload_optimizer_state=model_cfg.optimizer.offload,
            offload_param=model_cfg.offload,
            enable_bf16=model_cfg.enable_bf16,
            enable_fp16=model_cfg.enable_fp16,
        ),
    )


def make_inf_backend_config(model_cfg: ModelTrainEvalConfig, parallel_cfg: ParallelismConfig):
    if parallel_cfg.pipeline_parallel_size > 1:
        return ModelBackend("pipe_inference")
    else:
        return ModelBackend("null")


def make_model_config(cfg: ModelTrainEvalConfig):
    return get_real_model_config(
        model_path=cfg.path,
        hf_model_family=cfg.type._class,
        is_critic=cfg.type.is_critic,
        init_critic_from_actor=False,
        dtype="bf16" if cfg.enable_bf16 else "fp16",
        lora=cfg.lora,
    )


def resolve_rpc_hooks(rpc_allocs: List[RPCAllocation]):
    from reallm.api.quickstart.model import parallelism_config_equal

    for rpc_alloc in rpc_allocs:
        rpc = rpc_alloc.rpc
        parallel = rpc_alloc.parallel
        device_mesh = rpc_alloc.device_mesh
        # check param realloc hooks for train_step rpcs
        # only one param realloc is possible in each iteration
        if rpc.interface_type == ModelInterfaceType.TRAIN_STEP:
            for other in rpc_allocs:
                if rpc.name == other.rpc.name:
                    continue
                if rpc.model_name.role == other.rpc.model_name.role and not (parallelism_config_equal(
                        parallel, other.parallel) and device_mesh == other.device_mesh):
                    rpc.model_name = ModelName(rpc.model_name.role, other.rpc.model_name.replica_id + 1)
                    rpc.pre_hooks.append(SyncParamHook(source=other.rpc.model_name))
                    rpc.post_hooks.append(SyncParamHook(target=other.rpc.model_name))
                    break

        # add offload hooks for inference rpcs
        if rpc.interface_type == ModelInterfaceType.INFERENCE:
            offload_flag = True
            # if there is training rpcs for the same model, can not offload
            for other in rpc_allocs:
                if (other.rpc.model_name.role == rpc.model_name.role
                        and other.rpc.interface_type == ModelInterfaceType.TRAIN_STEP):
                    offload_flag = False
            if offload_flag:
                rpc.post_hooks.append(OffloadHook())
