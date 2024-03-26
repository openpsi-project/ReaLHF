from typing import List
import dataclasses
import functools

import numpy as np

from profiler.utils import find_factors

from api.config.config_base import MODEL_TYPE_TO_PATH
from api.config.config_flash_model import (get_flash_mqat_model_config, ModelTrainEvalConfig, OptimizerConfig,
                                           ParallelismConfig)
from api.config.config_system import *
from api.config.dfg import ModelInterface, ModelInterfaceType, ModelRPC, ModelType
from base.topology import PipeModelDataParallelTopology
from experiments.autoexp.device_mapping import (_make_inf_backend_config, _make_train_backend_config,
                                                RPCAllocation)

NUM_GPUS_PER_NODE = 8


# experiment config to run profiler (single instruction and model rpc)
@dataclasses.dataclass
class ProfileExperiment(Experiment):
    model_type: ModelType
    interface: ModelInterface

    n_nodes: int
    nodelist: str
    parallelism_config: ParallelismConfig

    seed: int = 1
    device_mesh_name: Optional[str] = None

    # list for profiler to enumerate
    bs_list: Optional[List[int]] = None
    seq_len_list: Optional[List[int]] = None
    gen_tokens_list: Optional[List[int]] = None

    # use_sequence_parallel: bool = False
    use_gradient_checkpointing: bool = False

    profile_communication: bool = False
    profile_rpc: bool = True

    def __post_init__(self):
        self.n_workers = self.n_nodes * NUM_GPUS_PER_NODE
        self.device_mesh_name = self.nodelist

    @property
    def rpcs(self) -> List[ModelRPC]:
        rollout = ModelRPC(model_name="actor",
                           model_type=self.model_type,
                           interface_type=ModelInterfaceType.GENERATE,
                           interface_impl=self.interface,
                           min_n_seqs=32,
                           max_n_seqs=32,
                           max_n_tokens=32 * 128)

        inf = ModelRPC(model_name="actor",
                       model_type=self.model_type,
                       interface_type=ModelInterfaceType.INFERENCE,
                       interface_impl=self.interface,
                       min_n_seqs=32,
                       max_n_seqs=32,
                       max_n_tokens=32 * 128)

        train = ModelRPC(model_name="actor",
                         model_type=self.model_type,
                         interface_type=ModelInterfaceType.TRAIN_STEP,
                         interface_impl=self.interface,
                         min_n_seqs=32,
                         max_n_seqs=32,
                         max_n_tokens=32 * 128)

        return [rollout, inf, train]

    @property
    def rpc_allocations(self):
        rollout, inf, train = self.rpcs
        return [
            RPCAllocation(
                rpc=rollout,
                mapping=np.ones((self.n_nodes, NUM_GPUS_PER_NODE), dtype=np.int32),
                train_eval_config=ModelTrainEvalConfig(
                    type="llama",
                    path=MODEL_TYPE_TO_PATH[rollout.model_type],
                    base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                    parallel=self.parallelism_config,
                ),
            ),
            RPCAllocation(
                rpc=inf,
                mapping=np.ones((self.n_nodes, NUM_GPUS_PER_NODE), dtype=np.int32),
                train_eval_config=ModelTrainEvalConfig(
                    type="llama",
                    path=MODEL_TYPE_TO_PATH[rollout.model_type],
                    base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                    parallel=self.parallelism_config,
                ),
            ),
            RPCAllocation(
                rpc=train,
                mapping=np.ones((self.n_nodes, NUM_GPUS_PER_NODE), dtype=np.int32),
                train_eval_config=ModelTrainEvalConfig(
                    type="llama",
                    path=MODEL_TYPE_TO_PATH[rollout.model_type],
                    base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                    gradient_checkpointing=self.use_gradient_checkpointing,
                    parallel=self.parallelism_config,
                    optimizer=OptimizerConfig(type="adam"),
                ),
            ),
        ]

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(profile_worker=TasksGroup(count=self.n_workers,
                                                              scheduling=Scheduling.profile_worker_default(
                                                                  cpu=4,
                                                                  gpu=1,
                                                                  gpu_type="tesla",
                                                                  mem=100000,
                                                                  nodelist=self.nodelist,
                                                              )))

    def initial_setup(self) -> ExperimentConfig:
        exp_ctrl: ExperimentSaveEvalControl = dataclasses.field(default_factory=functools.partial(
            ExperimentSaveEvalControl,
            benchmark_steps=20,
        ),)

        # rpc allocation for each rpc
        rpc = self.rpcs[0]
        m = self.rpc_allocations[0]
        model = Model(
            "flash_mqat",
            args=dict(
                model_path=MODEL_TYPE_TO_PATH[rpc.model_type],
                from_type="hf_as_actor",
                dtype="fp16",
                hf_model_type=rpc.model_type._class,
                tokenizer_path=MODEL_TYPE_TO_PATH[rpc.model_type],
                sequence_parallel=m.topo.get_dim("model") > 1,
                gradient_accumulation_fusion=False,
            ),
        )
        interface = rpc.interface_impl
        backend = _make_train_backend_config(
            m.train_eval_config,
            use_stream_pipe_engine=False,
        )

        profile_workers = [
            ProfileWorker(seed=self.seed,
                          model=model,
                          backend=backend,
                          interface=interface,
                          rpcs=self.rpcs,
                          topo=PipeModelDataParallelTopology(
                              num_dp=self.parallelism_config.data_parallel_size,
                              num_mp=self.parallelism_config.model_parallel_size,
                              num_pp=self.parallelism_config.pipeline_parallel_size,
                          ),
                          bs_list=self.bs_list,
                          seq_len_list=self.seq_len_list,
                          gen_tokens_list=self.gen_tokens_list,
                          profile_communication=False,
                          profile_rpc=True,
                          warmup_rounds=2,
                          profile_rounds=5) for _ in range(self.n_workers)
        ]

        return ExperimentConfig(exp_ctrl=exp_ctrl,
                                model_rpcs=[],
                                model_worker=[],
                                profile_worker=profile_workers)


def register_profile_experiment(
    size: int,
    num_pp: int,
    num_mp: int,
    num_dp: int,
):
    assert size in [7, 13, 34, 70]
    model_class = "llama" if size != 34 else "codellama"
    actor_model_type = ModelType(model_class, size, False)
    n_nodes = (num_pp * num_mp * num_dp) // NUM_GPUS_PER_NODE

    node_start = 40
    node_end = 40 + n_nodes - 1
    nodelist = f"QH-com[{node_start:02d}-{node_end:02d}]"

    exp_func = functools.partial(
        ProfileExperiment,
        model_type=actor_model_type,
        interface=ModelInterface(type_="profile"),
        n_nodes=n_nodes,
        nodelist=nodelist,
        parallelism_config=ParallelismConfig(
            data_parallel_size=num_dp,
            model_parallel_size=num_mp,
            pipeline_parallel_size=num_pp,
            use_sequence_parallel=(num_mp > 1),
        ),
    )
    # print(f"registering profile-s{size}p{num_pp}m{num_mp}d{num_dp}")
    register_experiment(f"profile-s{size}p{num_pp}m{num_mp}d{num_dp}", exp_func)


# register_profile_experiment(7, 2, 1, 4)

for size in [7, 13, 34, 70]:
    if size == 7:
        n_nodes = 1
    elif size == 13:
        n_nodes = 2
    elif size == 34:
        n_nodes = 4
    elif size == 70:
        n_nodes = 8

    num_gpus = n_nodes * NUM_GPUS_PER_NODE
    for num_mp in [1, 2, 4, 8]:
        remain = num_gpus // num_mp
        for num_dp in find_factors(remain):
            num_pp = remain // num_dp
            register_profile_experiment(size, num_pp, num_mp, num_dp)
