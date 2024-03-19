from collections import defaultdict
from typing import *
import dataclasses
import math
import re
import subprocess

import numpy as np
import transformers

from api.config.config_dataset import (
    PairedComparisonDatasetConfig,
    PromptAnswerDatasetConfig,
    PromptOnlyDatasetConfig,
)
from api.config.config_flash_model import (
    FLASH_MODEL_CONFIG_CONVERTER,
    FlashMQATConfig,
    MODEL_TYPE_TO_PATH,
    ModelTrainEvalConfig,
    ParallelismConfig,
    OptimizerConfig,
)
from api.config.config_system import *
from api.config.dfg import *
from base.topology import PipeModelDataParallelTopology
import base.logging as logging

logger = logging.getLogger("DeviceMappingCompiler", "benchmark")


@dataclasses.dataclass
class ClusterDeviceMesh:
    n_nodes: int
    n_gpus_per_node: int
    mem: Union[int, float]


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


def _are_ones_contiguous(binary_array: np.ndarray):
    one_indices = np.where(binary_array == 1)[0]
    if len(one_indices) == 0:
        return False
    return np.all(np.diff(one_indices) == 1)


def _is_valid_mapping(mapping: np.ndarray, device_mesh: ClusterDeviceMesh) -> bool:
    if mapping.shape != (device_mesh.n_nodes, device_mesh.n_gpus_per_node):
        raise RuntimeError(f"Invalid mapping shape {mapping} {device_mesh}")
    if not np.all(np.logical_or(mapping == 0, mapping == 1)):
        raise RuntimeError(f"Invalid mapping value {mapping}")

    assert math.log(device_mesh.n_gpus_per_node, 2).is_integer()

    one_node_valid_gpus = [2**i for i in range(int(math.log(device_mesh.n_gpus_per_node, 2)))]
    if mapping.sum() < device_mesh.n_gpus_per_node:
        if not any(mapping.sum() == g for g in one_node_valid_gpus):
            raise RuntimeError(f"Invalid mapping sum {mapping}")
    else:
        if not (
            mapping.sum() % device_mesh.n_gpus_per_node == 0
            and np.all(np.logical_or(mapping.sum(1) == device_mesh.n_gpus_per_node, mapping.sum(1) == 0))
        ):
            raise RuntimeError(f"Invalid mapping sum {mapping}")
    if not _are_ones_contiguous(mapping.flatten()):
        raise RuntimeError(f"mapping devices are not contiguous {mapping}")
    return True


def _group_models_by_node(allocations: Dict[str, RPCAllocation]) -> Dict[Tuple[int, int], List[str]]:
    """Group models by the nodes they are allocated to.

    Args:
        allocations (Dict[str, RPCAllocation]): RPC allocations derived by auto device mapping.

    Returns:
        Dict[Tuple[int, int], List[str]]: (start, end) node index -> model names on these nodes.
    """
    node2models: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for m in allocations.values():
        node_mapping = m.mapping.any(1)
        node_idx_start, node_idx_end = np.where(node_mapping)[0][[0, -1]]

        k, v = None, None
        for st, ed in node2models.keys():
            s = set(range(st, ed + 1))
            if s.intersection(range(node_idx_start, node_idx_end + 1)):
                v = node2models.pop((st, ed))
                if m.rpc.model_name not in v:
                    v.append(m.rpc.model_name)
                ss = s.union(range(node_idx_start, node_idx_end + 1))
                k = (min(ss), max(ss))
                break
        if k is None:
            k = (node_idx_start, node_idx_end)
            v = [m.rpc.model_name]
        node2models[k] = v
    return node2models


def _slurm_hostname_key(hostname):
    """
    Custom sorting key function to sort Slurm hostnames.
    """
    # Extract node number from hostname
    match = re.match(r"(\D+)(\d+)", hostname)
    if match:
        prefix, number = match.groups()
        return (prefix, int(number))
    else:
        return (hostname,)


def scheduling_config_from_allocations(
    allocations: Dict[str, RPCAllocation],
    device_mesh: ClusterDeviceMesh,
    nodelist: Optional[str] = None,
) -> ExperimentScheduling:
    if nodelist is not None:
        try:
            hostnames: List[str] = (
                subprocess.check_output(
                    [
                        "scontrol",
                        "show",
                        "hostnames",
                        nodelist,
                    ]
                )
                .decode("utf-8")
                .strip()
                .split("\n")
            )
            assert len(hostnames) == device_mesh.n_nodes
            hostnames = sorted(hostnames, key=_slurm_hostname_key)
        except FileNotFoundError:
            hostnames = None
            logger.warning(
                "scontrol not found, nodelist will be ignored. " "You are probably running in the local mode."
            )

    assert all(_is_valid_mapping(m.mapping, device_mesh) for m in allocations.values())
    node2models = _group_models_by_node({m.rpc.name: m for m in allocations.values()})

    sched = ExperimentScheduling(
        master_worker=TasksGroup(
            count=1,
            scheduling=Scheduling.master_worker_default(
                cpu=16,
                mem=20000,
                nodelist=nodelist,
            ),
        ),
        model_worker=[],
    )
    for st, ed in node2models:
        if nodelist is not None and hostnames is not None:
            _this_nodelist = ",".join(hostnames[st : ed + 1])
        else:
            _this_nodelist = None
        node_count = ed - st + 1
        sched.model_worker.append(
            TasksGroup(
                count=node_count * device_mesh.n_gpus_per_node,
                scheduling=Scheduling.model_worker_default(
                    cpu=4,
                    gpu=1,
                    gpu_type="tesla",
                    mem=100000,
                    nodelist=_this_nodelist,
                ),
            ),
        )
    return sched


def _make_train_backend_config(cfg: ModelTrainEvalConfig, use_stream_pipe_engine: bool):
    if cfg.parallel.pipeline_parallel_size > 1:
        engine_type = "stream_pipe" if use_stream_pipe_engine else "pipe"
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
            zero_stage=(
                cfg.zero_stage if cfg.parallel.pipeline_parallel_size == 1 else min(cfg.zero_stage, 1)
            ),
            gradient_checkpointing=cfg.gradient_checkpointing,
            engine_type=engine_type,
            offload_optimizer_state=cfg.optimizer.offload,
            offload_param=cfg.offload,
            enable_bf16=cfg.enable_bf16,
            enable_fp16=cfg.enable_fp16,
            sequence_parallel=cfg.parallel.use_sequence_parallel,
            enable_async_p2p_communication=cfg.enable_async_p2p,
        ),
    )


def _make_inf_backend_config(cfg: ModelTrainEvalConfig):
    return ModelBackend(
        "ds_inference",
        args=dict(
            enable_fp16=(not cfg.enable_bf16),
            zero_stage=3 if cfg.offload else 0,
            offload=cfg.offload,
            enable_bf16=cfg.enable_bf16,
            engine_type="pipe" if cfg.parallel.pipeline_parallel_size > 1 else "deepspeed",
            sequence_parallel=cfg.parallel.use_sequence_parallel,
        ),
    )


def mw_config_from_allocations(
    allocations: Dict[str, RPCAllocation],
    model_configs: Dict[str, Model],
    device_mesh: ClusterDeviceMesh,
    seed: int = 42,
    datasets: Optional[List[Dataset]] = None,
    dataloader: Optional[DataLoader] = None,
    tokenizer_path: Optional[str] = None,
) -> List[ModelWorker]:
    mw_configs = []
    shard_counter = defaultdict(lambda: 0)
    for i in range(device_mesh.n_nodes):
        for j in range(device_mesh.n_gpus_per_node):
            mw = ModelWorker(
                seed=seed,
                shards=[],
                cuda_cache_cleanliness=True,
                datasets=datasets,
                dataloader=dataloader,
                tokenizer_name_or_path=tokenizer_path,
            )
            for m in allocations.values():
                if m.mapping[i, j] and not any(m.rpc.model_name == s.id.model_name for s in mw.shards):
                    shard_idx = shard_counter[m.rpc.model_name]
                    if m.train_eval_config.optimizer.type != "empty":
                        backend = _make_train_backend_config(
                            m.train_eval_config,
                            use_stream_pipe_engine=False,
                        )
                    else:
                        backend = _make_inf_backend_config(m.train_eval_config)
                    mw.shards.append(
                        StandaloneModelShard(
                            id=ModelShardID(
                                # TODO: change this to rpc name instead of model name
                                model_name=m.rpc.model_name,
                                topo=m.topo,
                                dp_rank=m.topo.get_coord(shard_idx).data,
                                pp_rank=m.topo.get_coord(shard_idx).pipe,
                                mp_rank=m.topo.get_coord(shard_idx).model,
                            ),
                            model=model_configs[m.rpc.model_name],
                            backend=backend,
                        )
                    )
                    shard_counter[m.rpc.model_name] += 1
            mw_configs.append(mw)
    return mw_configs


def optimal_device_mapping(
    device_mesh: ClusterDeviceMesh,
    model_rpcs: List[ModelRPC],
    model_configs: Dict[str, FlashMQATConfig],
    nodelist: Optional[str] = None,
) -> Dict[str, RPCAllocation]:
    # NOTE: here we return model RPCs because different RPCs of the same model
    # may be assigned to different devices, thus have diferent model names.

    # HACK
    rollout, rew_inf, ref_inf, critic_inf, actor_train, critic_train = model_rpcs
    return {
        rollout.name: RPCAllocation(
            rpc=rollout,
            mapping=np.array([[1, 1, 1, 1, 0, 0, 0, 0]]),
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=MODEL_TYPE_TO_PATH[rollout.model_type],
                base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                gradient_checkpointing=True,
                parallel=ParallelismConfig(
                    model_parallel_size=2,
                    pipeline_parallel_size=2,
                    data_parallel_size=1,
                    use_sequence_parallel=True,
                ),
                optimizer=OptimizerConfig(type="adam"),
            ),
        ),
        rew_inf.name: RPCAllocation(
            rpc=rew_inf,
            mapping=np.array([[1, 1, 1, 1, 0, 0, 0, 0]]),
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=MODEL_TYPE_TO_PATH[rollout.model_type],
                base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                parallel=ParallelismConfig(
                    model_parallel_size=1,
                    pipeline_parallel_size=4,
                    data_parallel_size=1,
                    use_sequence_parallel=True,
                ),
            ),
        ),
        ref_inf.name: RPCAllocation(
            rpc=ref_inf,
            mapping=np.array([[0, 0, 0, 0, 1, 1, 1, 1]]),
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=MODEL_TYPE_TO_PATH[rollout.model_type],
                base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                parallel=ParallelismConfig(
                    model_parallel_size=2,
                    pipeline_parallel_size=1,
                    data_parallel_size=2,
                    use_sequence_parallel=True,
                ),
            ),
        ),
        critic_inf.name: RPCAllocation(
            rpc=critic_inf,
            mapping=np.array([[0, 0, 0, 0, 1, 1, 1, 1]]),
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=MODEL_TYPE_TO_PATH[rollout.model_type],
                base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                gradient_checkpointing=True,
                parallel=ParallelismConfig(
                    model_parallel_size=1,
                    pipeline_parallel_size=2,
                    data_parallel_size=2,
                    use_sequence_parallel=True,
                ),
                optimizer=OptimizerConfig(type="adam"),
            ),
        ),
        critic_train.name: RPCAllocation(
            rpc=critic_train,
            mapping=np.array([[0, 0, 0, 0, 1, 1, 1, 1]]),
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=MODEL_TYPE_TO_PATH[rollout.model_type],
                base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                gradient_checkpointing=True,
                parallel=ParallelismConfig(
                    model_parallel_size=1,
                    pipeline_parallel_size=2,
                    data_parallel_size=2,
                    use_sequence_parallel=True,
                ),
                optimizer=OptimizerConfig(type="adam"),
            ),
        ),
        actor_train.name: RPCAllocation(
            rpc=actor_train,
            mapping=np.array([[1, 1, 1, 1, 0, 0, 0, 0]]),
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=MODEL_TYPE_TO_PATH[rollout.model_type],
                base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                gradient_checkpointing=True,
                parallel=ParallelismConfig(
                    model_parallel_size=2,
                    pipeline_parallel_size=2,
                    data_parallel_size=1,
                    use_sequence_parallel=True,
                ),
                optimizer=OptimizerConfig(type="adam"),
            ),
        ),
    }


def auto_device_mapping(device_mesh: ClusterDeviceMesh, nodelist: Optional[str] = None):

    def _auto_device_mapping(
        experiment_cls: Type,
    ) -> Type[Experiment]:

        class AutoMappedExperiment:

            def __init__(self, *args, **kwargs):
                self._internal_exp = experiment_cls(*args, **kwargs)
                assert hasattr(self._internal_exp, "rpcs")
                assert all(isinstance(rpc, ModelRPC) for rpc in self._internal_exp.rpcs)

                model_configs = {}
                for rpc in self._internal_exp.rpcs:
                    path = MODEL_TYPE_TO_PATH[rpc.model_type]
                    hf_config = transformers.AutoConfig.from_pretrained(os.path.join(path, "config.json"))
                    config = FLASH_MODEL_CONFIG_CONVERTER[rpc.model_type._class](hf_config)
                    if rpc.model_name not in model_configs:
                        model_configs[rpc.model_name] = config
                    else:
                        for k, v in dataclasses.asdict(config).items():
                            if getattr(model_configs[rpc.model_name], k) != v:
                                raise RuntimeError(
                                    f"Model config mismatch: {k} {v} {getattr(model_configs[rpc.model_name], k)}"
                                )
                self._rpcs: List[ModelRPC] = self._internal_exp.rpcs
                self._allocations = optimal_device_mapping(
                    device_mesh,
                    model_rpcs=self._internal_exp.rpcs,
                    model_configs=model_configs,
                    nodelist=nodelist,
                )

            def scheduling_setup(self):
                return scheduling_config_from_allocations(
                    self._allocations,
                    device_mesh=device_mesh,
                    nodelist=nodelist,
                )

            def initial_setup(self):
                model_configs: Dict[str, Model] = {}
                for rpc in self._rpcs:
                    if rpc.model_name in model_configs:
                        continue
                    m: RPCAllocation = self._allocations[rpc.name]
                    path = MODEL_TYPE_TO_PATH[rpc.model_type]
                    model_configs[rpc.model_name] = Model(
                        "flash_mqat",
                        args=dict(
                            model_path=path,
                            from_type="hf_as_critic" if rpc.model_type.is_critic else "hf_as_actor",
                            dtype="fp16",
                            hf_model_type=rpc.model_type._class,
                            tokenizer_path=path,
                            sequence_parallel=m.topo.get_dim("model") > 1,
                            gradient_accumulation_fusion=False,
                        ),
                    )
                src_rpc: ModelRPC = [rpc for rpc in self._rpcs if rpc.is_src][0]
                tokenizer_path = MODEL_TYPE_TO_PATH[src_rpc.model_type]

                if isinstance(self._internal_exp.dataset, PromptOnlyDatasetConfig):
                    dataset = Dataset(
                        "packed_prompt",
                        args=dict(
                            dataset_path=self._internal_exp.dataset.path,
                            n_tokens_per_batch=self._internal_exp.dataset.n_tokens_per_batch,
                            max_length=self._internal_exp.dataset.max_prompt_len,
                        ),
                    )
                    dataloader = DataLoader("iterable_dataset_loader")
                else:
                    raise NotImplementedError()

                model_worker = mw_config_from_allocations(
                    self._allocations,
                    model_configs,
                    device_mesh,
                    datasets=[dataset],
                    dataloader=dataloader,
                    tokenizer_path=tokenizer_path,
                    seed=getattr(self._internal_exp, "seed", 1),
                )

                return ExperimentConfig(
                    exp_ctrl=getattr(
                        self._internal_exp.exp_ctrl,
                        "exp_ctrl",
                        ExperimentSaveEvalControl(benchmark_steps=20),
                    ),
                    model_worker=model_worker,
                    model_rpcs=self._rpcs,
                )

        return AutoMappedExperiment

    return _auto_device_mapping
