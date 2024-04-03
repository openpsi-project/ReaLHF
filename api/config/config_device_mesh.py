from typing import Union
import dataclasses

import numpy as np

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
