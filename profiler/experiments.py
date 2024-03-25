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
    min_n_seqs=256,
    max_n_seqs=257,
    max_n_tokens=256 * 256,  # generate 256 tokens
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
    min_n_seqs=256,
    max_n_seqs=257,
    max_n_tokens=512 * 256,
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
    min_n_seqs=256,
    max_n_seqs=257,
    max_n_tokens=512 * 256,
    max_concurrent_calls=1,
)

inf_values = ModelRPC(
    "critic",
    ModelInterfaceType.INFERENCE,
    input_data=["packed_seq", "cu_seqlens", "seq_no_eos_mask"],
    output_data=["scores"],
    output_key_remap={"scores": "values"},
    dp_broker_type="packed",
    min_n_seqs=256,
    max_n_seqs=257,
    max_n_tokens=512 * 256,
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
    min_n_seqs=256,
    max_n_seqs=257,
    max_n_tokens=512 * 256,
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
    min_n_seqs=256,
    max_n_seqs=257,
    max_n_tokens=512 * 256,
    max_concurrent_calls=1,
)

NUM_GPUS_PER_NODE = 8
LLAMA_2_7B_PATH = "/lustre/public/pretrained_model_weights/sharded/Llama-2-7b-hf_4pp_3s"
LLAMA_2_13B_PATH = "/lustre/public/pretrained_model_weights/sharded/Llama-2-13b-hf_4mp_3s"
LLAMA_2_70B_PATH = "/lustre/public/pretrained_model_weights/sharded/Llama-2-70b-hf_8pp_3s"


@dataclasses.dataclass
class ProfileExperiment(Experiment):
    model_paths: Optional[List[str]] = None
    model_types: Optional[List[str]] = None
    model_names: Optional[List[str]] = None
    model_rpcs: Optional[List[ModelRPC]] = None
    model_names_to_types: Optional[Dict[str, str]] = None

    seed: int = 1
    n_nodes: int = 2
    nodelist: str = "QH-com[17-18]"
    device_mesh_name: str = "QH-com[17-18]"

    num_pp: int = 2
    num_mp: int = 1
    num_dp: int = 8

    profile_communication: bool = False
    profile_model_function_call: bool = True
    profile_mfc_path: str = LLAMA_2_13B_PATH

    use_sequence_parallel: bool = False
    use_gradient_checkpointing: bool = False

    def __post_init__(self):
        self.n_workers = self.n_nodes * NUM_GPUS_PER_NODE
        assert self.num_dp * self.num_pp * self.num_mp == self.n_workers

        if self.model_paths is None:
            # example
            self.model_paths = [LLAMA_2_13B_PATH, LLAMA_2_7B_PATH]
            self.model_types = ["Llama-2-13b", "Llama-2-7b"]
            self.model_names = ["actor", "reward", "ref", "critic"]
            self.model_names_to_types = {
                "actor": "Llama-2-13b",
                "reward": "Llama-2-7b",
                "ref": "Llama-2-13b",
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
            num_pp=self.num_pp,
            num_mp=self.num_mp,
            num_dp=self.num_dp,
        )

        model = get_flash_mqat_model_config(
            from_type="self",
            model_path=self.profile_mfc_path,
            hf_model_type="llama",
            tokenizer_path=self.profile_mfc_path,
            use_pipe=True,
            dtype="fp16",
            sequence_parallel=self.use_sequence_parallel,
            partition_method="parameters_balanced",
            lora=None,
        )

        backend = ModelBackend(
            type_="ds_train",
            args=dict(
                optimizer_name="adam",
                optimizer_config=dict(lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95)),
                warmup_steps_proportion=0.0,
                min_lr_ratio=0.0,
                zero_stage=1,
                engine_type="profile",
                gradient_checkpointing=self.use_gradient_checkpointing,
                num_pipeline_stages=self.num_pp,
                enable_fp16=True,
                enable_bf16=False,
                sequence_parallel=self.use_sequence_parallel,
                enable_async_p2p_communication=False,
            ),
        )

        interface = ModelInterface(type_="profile", args=dict())

        profile_workers = [
            ProfileWorker(
                seed=self.seed,
                model=model,
                backend=backend,
                interface=interface,
                model_name="profile",
                device_mesh_name=self.device_mesh_name,
                topo=topo,
                dp_rank=topo.get_coord(i).data,
                pp_rank=topo.get_coord(i).pipe,
                mp_rank=topo.get_coord(i).model,
                cuda_cache_cleanliness=True,
                profile_communication=self.profile_communication,
                profile_model_function_call=self.profile_model_function_call,
            ) for i in range(self.n_workers)
        ]

        return ExperimentConfig(total_train_epochs=1,
                                model_rpcs=[],
                                data_worker=[],
                                model_worker=[],
                                profile_worker=profile_workers)


register_experiment("profile", ProfileExperiment)
