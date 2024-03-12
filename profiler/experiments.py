from typing import List
import dataclasses

from api.config import *
from api.config import ExperimentScheduling
from api.dfg import ModelInterfaceType, ModelRPC
from base.topology import PipeModelDataParallelTopology
from experiments.common.config_model import get_flash_mqat_model_config

# currently only support max_n_seqs=min_n_seqs+1,
# i.e. fixed batch size for each model function call
# max_n_tokens = min_n_tokens * max_seq_len
rollout = ModelRPC(
    "actor",
    ModelInterfaceType.GENERATE,
    input_data=["packed_prompts", "prompt_cu_seqlens"],
    output_data=[
        "seq_no_eos_mask",
        "packed_seq",
        "cu_seqlens",
        "packed_logprobs",
        "packed_logits_mask",
        "prompt_mask",
    ],
    dp_broker_type="packed",
    min_n_seqs=32,
    max_n_seqs=33,
    max_n_tokens=128 * 32,  # generate 256 tokens
    max_concurrent_calls=4,
)

inf_reward = ModelRPC(
    "reward",
    ModelInterfaceType.INFERENCE,
    input_data=["packed_seq", "cu_seqlens"],
    input_key_remap={"packed_seq": "packed_input_ids"},
    output_data=["scores"],
    output_key_remap={"scores": "rewards"},
    dp_broker_type="packed",
    min_n_seqs=32,
    max_n_seqs=33,
    max_n_tokens=256 * 32,
    max_concurrent_calls=1,
)

inf_ref_logits = ModelRPC(
    "ref",
    ModelInterfaceType.INFERENCE,
    input_data=[
        "packed_seq",
        "cu_seqlens",
        "packed_logits_mask",
    ],
    output_data=["logprobs"],
    output_key_remap={"logprobs": "packed_ref_logprobs"},
    dp_broker_type="packed",
    min_n_seqs=32,
    max_n_seqs=33,
    max_n_tokens=256 * 32,
    max_concurrent_calls=1,
)

inf_values = ModelRPC(
    "critic",
    ModelInterfaceType.INFERENCE,
    input_data=["packed_seq", "cu_seqlens", "seq_no_eos_mask"],
    output_data=["scores"],
    output_key_remap={"scores": "values"},
    dp_broker_type="packed",
    min_n_seqs=32,
    max_n_seqs=33,
    max_n_tokens=256 * 32,
    max_concurrent_calls=1,
)

train_actor = ModelRPC(
    "actor",
    ModelInterfaceType.TRAIN_STEP,
    input_data=[
        "packed_seq",
        "cu_seqlens",
        "packed_logprobs",
        "packed_ref_logprobs",
        "rewards",
        "values",
        "prompt_mask",
        "seq_no_eos_mask",
        "packed_logits_mask",
    ],
    log_return_value=True,
    dp_broker_type="packed",
    min_n_seqs=32,
    max_n_seqs=33,
    max_n_tokens=256 * 32,
    max_concurrent_calls=1,
)

train_critic = ModelRPC(
    "critic",
    ModelInterfaceType.TRAIN_STEP,
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
    dp_broker_type="packed",
    log_return_value=True,
    min_n_seqs=32,
    max_n_seqs=33,
    max_n_tokens=256 * 32,
    max_concurrent_calls=1,
)

NUM_GPUS_PER_NODE = 8
LLAMA_2_7B_PATH = "/lustre/public/pretrained_model_weights/sharded/Llama-2-7b-hf_4pp_3s"


@dataclasses.dataclass
class ProfileExperiment(Experiment):
    model_paths: Optional[List[str]] = None
    model_types: Optional[List[str]] = None
    model_names: Optional[List[str]] = None
    model_rpcs: Optional[List[ModelRPC]] = None
    model_names_to_types: Optional[Dict[str, str]] = None

    seed: int = 1
    n_nodes: int = 4
    nodelist: str = "QH-com[40-43]"
    device_mesh_name: str = "QH-com[40-43]"

    def __post_init__(self):
        self.n_workers = self.n_nodes * NUM_GPUS_PER_NODE

        if self.model_paths is None:
            # example
            self.model_paths = [LLAMA_2_7B_PATH]
            self.model_types = ["Llama-2-7b"]
            self.model_names = ["actor", "reward", "ref", "critic"]
            self.model_names_to_types = {
                "actor": "Llama-2-7b",
                "reward": "Llama-2-7b",
                "ref": "Llama-2-7b",
                "critic": "Llama-2-7b",
            }
            self.model_rpcs = [rollout, inf_reward, inf_ref_logits, inf_values, train_actor, train_critic]
        else:
            assert self.model_types is not None
            assert self.model_names is not None
            assert self.model_rpcs is not None
            assert len(self.model_paths) == len(self.model_types)

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
        topo = PipeModelDataParallelTopology(
            num_pp=1,
            num_mp=1,
            num_dp=self.n_workers,
        )

        profile_workers = [
            ProfileWorker(
                seed=self.seed,
                model=None,
                backend=None,
                interface=None,
                model_name="profile",
                device_mesh_name=self.device_mesh_name,
                topo=topo,
                dp_rank=topo.get_coord(i).data,
                pp_rank=topo.get_coord(i).pipe,
                mp_rank=topo.get_coord(i).model,
                cuda_cache_cleanliness=True,
            ) for i in range(self.n_workers)
        ]

        return ExperimentConfig(total_train_epochs=1,
                                model_rpcs=[],
                                data_worker=[],
                                model_worker=[],
                                profile_worker=profile_workers)


register_experiment("profile", ProfileExperiment)
