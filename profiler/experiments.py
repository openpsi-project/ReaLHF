from typing import List
import dataclasses
import functools

import numpy as np

from profiler.utils import find_factors

from api.config.config_base import MODEL_TYPE_TO_PATH
from api.config.config_device_mesh import make_train_backend_config, RPCAllocation
from api.config.config_flash_model import ModelTrainEvalConfig, OptimizerConfig, ParallelismConfig
from api.config.config_system import *
from api.config.dfg import ModelInterface, ModelInterfaceType, ModelRPC, ModelType
from base.topology import PipeModelDataParallelTopology

NUM_GPUS_PER_NODE = 8


def ppo_rpcs_example(size):
    gen_bs = train_bs = 256
    seq_len = 256
    gen_len = 256
    actor_interface = ModelInterface(
        "flash_actor",
        args={},
    )
    ref_interface = copy.deepcopy(actor_interface)
    ref_interface.args["enable_save"] = False

    critic_interface = ModelInterface(
        "flash_critic",
        args={},
    )
    rw_interface = ModelInterface(
        "flash_paired_rw",
        args=dict(enable_save=False,),
    )
    model_class = "llama" if size != 34 else "codellama"
    return [
        ModelRPC(
            model_name="actor",
            model_type=ModelType(model_class, size, is_critic=False),
            interface_type=ModelInterfaceType.GENERATE,
            interface_impl=actor_interface,
            input_data=["packed_prompts", "prompt_cu_seqlens"],
            output_data=[
                "seq_no_eos_mask",
                "packed_seq",
                "cu_seqlens",
                "packed_logprobs",
                "prompt_mask",
            ],
            balanced_dp=True,
            min_n_seqs=gen_bs,
            max_n_seqs=gen_bs,
            max_n_tokens=gen_bs * seq_len,
        ),
        ModelRPC(
            model_name="reward",
            model_type=ModelType("llama", 7, is_critic=True),
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=rw_interface,
            input_data=["packed_seq", "cu_seqlens"],
            input_key_remap={"packed_seq": "packed_input_ids"},
            output_data=["scores"],
            output_key_remap={"scores": "rewards"},
            min_n_seqs=gen_bs,
            max_n_seqs=gen_bs,
            max_n_tokens=gen_bs * (seq_len + gen_len),
        ),
        ModelRPC(
            model_name="ref",
            model_type=ModelType(model_class, size, is_critic=False),
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=ref_interface,
            input_data=[
                "packed_seq",
                "cu_seqlens",
            ],
            output_data=["logprobs"],
            output_key_remap={"logprobs": "packed_ref_logprobs"},
            min_n_seqs=gen_bs,
            max_n_seqs=gen_bs,
            max_n_tokens=gen_bs * (seq_len + gen_len),
        ),
        ModelRPC(
            model_name="critic",
            model_type=ModelType("llama", 7, is_critic=True),
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=critic_interface,
            input_data=["packed_seq", "cu_seqlens", "seq_no_eos_mask"],
            output_data=["scores"],
            output_key_remap={"scores": "values"},
            min_n_seqs=gen_bs,
            max_n_seqs=gen_bs,
            max_n_tokens=gen_bs * (seq_len + gen_len),
        ),
        ModelRPC(
            model_name="actor",
            model_type=ModelType(model_class, size, is_critic=False),
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=actor_interface,
            input_data=[
                "packed_seq",
                "cu_seqlens",
                "packed_logprobs",
                "packed_ref_logprobs",
                "rewards",
                "values",
                "prompt_mask",
                "seq_no_eos_mask",
            ],
            log_return_value=True,
            # min_n_seqs_per_dp=self.ppo.ppo_n_minibatches,
            min_n_seqs=train_bs,
            max_n_seqs=train_bs,
            balanced_dp=True,
            max_n_tokens=train_bs * (seq_len + gen_len),
        ),
        ModelRPC(
            model_name="critic",
            interface_type=ModelInterfaceType.TRAIN_STEP,
            model_type=ModelType("llama", 7, is_critic=True),
            interface_impl=critic_interface,
            input_data=[
                "packed_seq",
                "cu_seqlens",
                "packed_logprobs",
                "packed_ref_logprobs",
                "rewards",
                "values",
                "prompt_mask",
                "seq_no_eos_mask",
            ],
            log_return_value=True,
            # min_n_seqs_per_dp=self.ppo.ppo_n_minibatches,
            min_n_seqs=train_bs,
            max_n_seqs=train_bs,
            balanced_dp=True,
            max_n_tokens=train_bs * (seq_len + gen_len),
        ),
    ]


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
    use_gradient_checkpointing: bool = True

    # profile_communication: bool = False
    # profile_rpc: bool = True

    single_rpc_profile: Optional[str] = None
    instruction_sync: bool = False

    def __post_init__(self):
        self.n_workers = self.n_nodes * NUM_GPUS_PER_NODE
        self.device_mesh_name = self.nodelist

    @property
    def all_rpcs(self):
        rollout = ModelRPC(model_name="actor",
                           model_type=self.model_type,
                           interface_type=ModelInterfaceType.GENERATE,
                           interface_impl=self.interface,
                           min_n_seqs=256,
                           max_n_seqs=256,
                           max_n_tokens=256 * 256)

        inf = ModelRPC(model_name="actor",
                       model_type=self.model_type,
                       interface_type=ModelInterfaceType.INFERENCE,
                       interface_impl=self.interface,
                       min_n_seqs=256,
                       max_n_seqs=256,
                       max_n_tokens=256 * 256)

        train = ModelRPC(model_name="actor",
                         model_type=self.model_type,
                         interface_type=ModelInterfaceType.TRAIN_STEP,
                         interface_impl=self.interface,
                         min_n_seqs=256,
                         max_n_seqs=256,
                         max_n_tokens=256 * 256)
        return [rollout, inf, train]

    @property
    def rpcs(self) -> List[ModelRPC]:
        if self.single_rpc_profile == "gen":
            return [self.all_rpcs[0]]
        elif self.single_rpc_profile == "inf":
            return [self.all_rpcs[1]]
        elif self.single_rpc_profile == "train":
            return [self.all_rpcs[2]]
        else:
            return self.all_rpcs

    @property
    def rpc_allocations(self):
        rollout, inf, train = self.all_rpcs
        rollout_alloc = RPCAllocation(
            rpc=rollout,
            mapping=np.ones((self.n_nodes, NUM_GPUS_PER_NODE), dtype=np.int32),
            train_eval_config=ModelTrainEvalConfig(
                type=rollout.model_type._class,
                path=MODEL_TYPE_TO_PATH[rollout.model_type],
                base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                parallel=self.parallelism_config,
            ),
        )
        inf_alloc = RPCAllocation(
            rpc=inf,
            mapping=np.ones((self.n_nodes, NUM_GPUS_PER_NODE), dtype=np.int32),
            train_eval_config=ModelTrainEvalConfig(
                type=inf.model_type._class,
                path=MODEL_TYPE_TO_PATH[rollout.model_type],
                base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                parallel=self.parallelism_config,
            ),
        )
        train_alloc = RPCAllocation(
            rpc=train,
            mapping=np.ones((self.n_nodes, NUM_GPUS_PER_NODE), dtype=np.int32),
            train_eval_config=ModelTrainEvalConfig(
                type=train.model_type._class,
                path=MODEL_TYPE_TO_PATH[rollout.model_type],
                base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                gradient_checkpointing=self.use_gradient_checkpointing,
                parallel=self.parallelism_config,
                optimizer=OptimizerConfig(type="adam"),
            ),
        )
        if self.single_rpc_profile == "gen":
            return [rollout_alloc]
        elif self.single_rpc_profile == "inf":
            return [inf_alloc]
        elif self.single_rpc_profile == "train":
            return [train_alloc]
        else:
            return [rollout_alloc, inf_alloc, train_alloc]

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
            benchmark_steps=10,
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
        backend = make_train_backend_config(m.train_eval_config, instruction_sync=self.instruction_sync)

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

    # node_start = 42
    # node_end = node_start + n_nodes - 1
    # nodelist = f"QH-com[{node_start:02d}-{node_end:02d}]"
    if size == 7:
        nodelist = "QH-com30"
    elif size == 13:
        nodelist = "QH-com[42-43]"
    elif size == 34:
        nodelist = "QH-com[42-45]"
    elif size == 70:
        nodelist = "QH-com[29-30,42-47]"

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
        instruction_sync=False,
    )
    # print(f"registering profile-s{size}p{num_pp}m{num_mp}d{num_dp}")
    register_experiment(f"profile-s{size}p{num_pp}m{num_mp}d{num_dp}", exp_func)

    for func_name in ["gen", "train", "inf"]:
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
            single_rpc_profile=func_name,
            instruction_sync=True,
        )
        register_experiment(f"profile-s{size}p{num_pp}m{num_mp}d{num_dp}-{func_name}", exp_func)


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
