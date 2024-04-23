from typing import Optional, Union
import dataclasses

import numpy as np

from api.config.config_base import ModelBackend
from api.config.config_flash_model import ModelTrainEvalConfig
from api.config.dfg import ModelRPC
from base.topology import PipeModelDataParallelTopology


@dataclasses.dataclass
class RPCAllocation:
    rpc: ModelRPC
    mapping: np.ndarray  # a 2D binary array, shape (n_nodes, n_gpus_per_node)
    train_eval_config: ModelTrainEvalConfig

    @property
    def topo(self) -> PipeModelDataParallelTopology:
        return PipeModelDataParallelTopology(
            num_pp=self.train_eval_config.parallel.pipeline_parallel_size,
            num_mp=self.train_eval_config.parallel.model_parallel_size,
            num_dp=self.train_eval_config.parallel.data_parallel_size,
        )


@dataclasses.dataclass
class ClusterDeviceMesh:
    n_nodes: int
    n_gpus_per_node: int
    mem: Union[int, float]


def make_train_backend_config(cfg: ModelTrainEvalConfig, instruction_sync: bool = False):
    if cfg.parallel.pipeline_parallel_size > 1:
        engine_type = "pipe"
    else:
        engine_type = "deepspeed"
    return ModelBackend(
        "ds_train",
        args=dict(
            optimizer_name="adam",
            optimizer_config=dict(
                lr=cfg.optimizer.lr,
                weight_decay=cfg.optimizer.weight_decay,
                eps=cfg.optimizer.eps,
                betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
            ),
            lr_scheduler_type=cfg.optimizer.lr_scheduler_type,
            warmup_steps_proportion=cfg.optimizer.warmup_steps_proportion,
            min_lr_ratio=cfg.optimizer.min_lr_ratio,
            zero_stage=(cfg.zero_stage if cfg.parallel.pipeline_parallel_size == 1 else min(
                cfg.zero_stage, 1)),
            gradient_checkpointing=cfg.gradient_checkpointing,
            engine_type=engine_type,
            offload_optimizer_state=cfg.optimizer.offload,
            offload_param=cfg.offload,
            enable_bf16=cfg.enable_bf16,
            enable_fp16=cfg.enable_fp16,
            sequence_parallel=cfg.parallel.use_sequence_parallel,
            enable_async_p2p_communication=cfg.enable_async_p2p,
            instruction_sync=instruction_sync,
        ),
    )


def make_inf_backend_config(cfg: ModelTrainEvalConfig):
    if cfg.parallel.pipeline_parallel_size > 1:
        return ModelBackend("pipe_inference")
    else:
        return ModelBackend("null")
    # return ModelBackend(
    #     "ds_inference",
    #     args=dict(
    #         enable_fp16=(not cfg.enable_bf16),
    #         zero_stage=3 if cfg.offload else 0,
    #         offload=cfg.offload,
    #         enable_bf16=cfg.enable_bf16,
    #         engine_type="pipe" if cfg.parallel.pipeline_parallel_size > 1 else "deepspeed",
    #         sequence_parallel=cfg.parallel.use_sequence_parallel,
    #     ),
    # )
