import collections
import itertools
import re
from typing import *

import numpy as np

from realhf.api.core.config import (
    ModelBackendAbstraction,
    ModelInterfaceType,
    ModelName,
)
from realhf.api.core.dfg import OffloadHook, ParamReallocHook
from realhf.api.quickstart.device_mesh import RPCAllocation
from realhf.api.quickstart.model import (
    ModelTrainEvalConfig,
    ParallelismConfig,
    parallelism_eq,
)
from realhf.base import logging
from realhf.base.topology import PipeModelDataParallelTopology

logger = logging.getLogger("Experiment Common Utils", "benchmark")


def get_topo(
    parallel: ParallelismConfig,
    gradient_checkpointing: bool,
    gradient_accumulation_fusion: bool,
    max_prompt_len: Optional[int] = None,
) -> PipeModelDataParallelTopology:
    return PipeModelDataParallelTopology(
        num_mp=parallel.model_parallel_size,
        num_pp=parallel.pipeline_parallel_size,
        num_dp=parallel.data_parallel_size,
        sequence_parallel=parallel.use_sequence_parallel,
        gradient_checkpointing=gradient_checkpointing,
        max_prompt_len=max_prompt_len,
        gradient_accumulation_fusion=gradient_accumulation_fusion,
    )


def get_world_size(parallel: ParallelismConfig) -> int:
    return (
        parallel.model_parallel_size
        * parallel.pipeline_parallel_size
        * parallel.data_parallel_size
    )


def make_train_backend_config(
    model_cfg: ModelTrainEvalConfig, parallel_cfg: ParallelismConfig
):
    if model_cfg.backend == "deepspeed":
        return ModelBackendAbstraction(
            "deepspeed",
            args=dict(
                optimizer_name="adam",
                optimizer_config=dict(
                    lr=model_cfg.optimizer.lr,
                    weight_decay=model_cfg.optimizer.weight_decay,
                    eps=model_cfg.optimizer.eps,
                    betas=(
                        model_cfg.optimizer.beta1,
                        model_cfg.optimizer.beta2,
                    ),
                ),
                lr_scheduler_type=model_cfg.optimizer.lr_scheduler_type,
                warmup_steps_proportion=model_cfg.optimizer.warmup_steps_proportion,
                min_lr_ratio=model_cfg.optimizer.min_lr_ratio,
                zero_stage=(
                    model_cfg.zero_stage
                    if parallel_cfg.pipeline_parallel_size == 1
                    else min(model_cfg.zero_stage, 1)
                ),
                offload_optimizer_state=model_cfg.optimizer.offload,
                offload_param=model_cfg.offload,
                enable_bf16=model_cfg.enable_bf16,
                enable_fp16=model_cfg.enable_fp16,
            ),
        )
    elif model_cfg.backend == "megatron":
        if model_cfg.optimizer.offload or model_cfg.offload:
            raise ValueError("Offload is not supported in Megatron backend.")
        if model_cfg.zero_stage == 3:
            raise ValueError("Zero stage 3 is not supported in Megatron backend.")
        if model_cfg.zero_stage == 2:
            logger.warning(
                "Megatron does not support ZeRO stage 2. Degenerates to stage 1."
            )
            model_cfg.zero_stage = 1
        return ModelBackendAbstraction(
            "megatron",
            args=dict(
                optimizer_name="adam",
                optimizer_config=dict(
                    lr=model_cfg.optimizer.lr,
                    weight_decay=model_cfg.optimizer.weight_decay,
                    eps=model_cfg.optimizer.eps,
                    betas=(
                        model_cfg.optimizer.beta1,
                        model_cfg.optimizer.beta2,
                    ),
                ),
                lr_scheduler_type=model_cfg.optimizer.lr_scheduler_type,
                warmup_steps_proportion=model_cfg.optimizer.warmup_steps_proportion,
                min_lr_ratio=model_cfg.optimizer.min_lr_ratio,
                enable_bf16=model_cfg.enable_bf16,
                enable_fp16=model_cfg.enable_fp16,
                # See MegatronTrainBackend for detailed explanations about these options.
                use_zero_optimization=model_cfg.zero_stage > 0,
                overlap_grad_reduce=model_cfg.zero_stage > 0,
                overlap_param_gather=False,
            ),
        )
    else:
        raise NotImplementedError(f"Backend {model_cfg.backend} is not supported.")


def make_inf_backend_config(
    model_cfg: ModelTrainEvalConfig, parallel_cfg: ParallelismConfig
):
    return ModelBackendAbstraction("inference")


def resolve_replica_ids(rpc_allocs: List[RPCAllocation]):
    role_cnt = collections.defaultdict(int)
    first_device_mesh = dict()
    first_parallel = dict()
    for alloc in rpc_allocs:
        rpc = alloc.rpc
        if rpc.role not in first_device_mesh:
            first_device_mesh[rpc.role] = alloc.device_mesh
            first_parallel[rpc.role] = alloc.parallel
            continue
        if alloc.device_mesh != first_device_mesh[rpc.role] or not parallelism_eq(
            alloc.parallel, first_parallel[rpc.role]
        ):
            role_cnt[rpc.role] += 1
            rpc.model_name = ModelName(rpc.role, role_cnt[rpc.role])


def resolve_rpc_hooks(
    rpc_allocs: List[RPCAllocation], model_configs: Dict[str, ModelTrainEvalConfig]
):
    role_interface_types = collections.defaultdict(set)
    for rpc_alloc in rpc_allocs:
        role_interface_types[rpc_alloc.rpc.role].add(rpc_alloc.rpc.interface_type)

    for i, rpc_alloc in enumerate(rpc_allocs):
        rpc = rpc_alloc.rpc
        parallel = rpc_alloc.parallel
        device_mesh = rpc_alloc.device_mesh
        # check param realloc hooks for train_step rpcs
        if rpc.interface_type == ModelInterfaceType.TRAIN_STEP:
            for j, other in enumerate(rpc_allocs):
                if rpc.name == other.rpc.name:
                    continue
                if rpc.role != other.rpc.role:
                    continue
                if (
                    parallelism_eq(parallel, other.parallel)
                    and device_mesh == other.device_mesh
                ):
                    continue
                self_config = model_configs[rpc.model_name.role]
                other_config = model_configs[other.rpc.model_name.role]
                if (
                    self_config.backend == "deepspeed"
                    or other_config.backend == "deepspeed"
                ):
                    raise ValueError(
                        "Param realloc hooks are not supported in DeepSpeed backend."
                    )
                other.rpc.add_pre_hook(ParamReallocHook(source=rpc.model_name))
                other.rpc.add_post_hook(ParamReallocHook(target=rpc.model_name))
                logger.info(
                    f"Add param sync hooks between "
                    f"{rpc.name} and {other.rpc.name} for role {rpc.role}"
                )

        # Add offload hooks for inference and generate rpcs.
        # Add the offload hook only if the role will not be trained (e.g., reward model)
        # and its allocation is overlapped with at least one other RPCs.
        # As a result, a single inference/generate RPC will not be offloaded.
        overlapped_with_other = False
        for other in rpc_allocs:
            if rpc.name == other.rpc.name:
                continue
            if np.any(np.logical_and(other.device_mesh.mapping, device_mesh.mapping)):
                overlapped_with_other = True
                break
        if (
            ModelInterfaceType.TRAIN_STEP not in role_interface_types[rpc.role]
            and overlapped_with_other
        ):
            rpc.add_post_hook(OffloadHook())
            logger.info(f"Add offload hook for rpc {rpc.name} for role {rpc.role}")


def extract_symmetric_allocation(allocation_mode: str) -> Dict | None:
    for x, y, z in itertools.permutations(["d", "m", "p"]):
        pattern = rf"{x}(\d+){y}(\d+){z}(\d+)"
        m = re.match(pattern, allocation_mode)
        if not m:
            continue
        a, b, c = map(int, m.groups())
        return {
            x: a,
            y: b,
            z: c,
        }
