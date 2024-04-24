from collections import defaultdict
from typing import *
import dataclasses
import math
import re
import subprocess

import numpy as np
import transformers

from reallm.api.core.config import MODEL_TYPE_TO_PATH
from reallm.api.core.dfg import *
from reallm.api.core.system_api import *
from reallm.api.quickstart.dataset import PromptOnlyDatasetConfig
from reallm.api.quickstart.device_mesh import *
from reallm.api.quickstart.model import (FLASH_MODEL_CONFIG_CONVERTER, ModelTrainEvalConfig, OptimizerConfig,
                                         ParallelismConfig, ReaLModelConfig)
from reallm.base.topology import PipeModelDataParallelTopology
from reallm.profiler.search import (data_pipe_device_mapping, full_model_device_mapping,
                                    model_pipe_device_mapping, optimal_device_mapping,
                                    test_model_device_mapping)
import reallm.base.logging as logging

logger = logging.getLogger("DeviceMappingCompiler", "benchmark")


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
        if not (mapping.sum() % device_mesh.n_gpus_per_node == 0
                and np.all(np.logical_or(mapping.sum(1) == device_mesh.n_gpus_per_node,
                                         mapping.sum(1) == 0))):
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
        # print(m.mapping)
        node_mapping = m.mapping.any(1)
        # print(node_mapping)
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
            hostnames: List[str] = (subprocess.check_output([
                "scontrol",
                "show",
                "hostnames",
                nodelist,
            ]).decode("utf-8").strip().split("\n"))
            assert len(hostnames) == device_mesh.n_nodes
            hostnames = sorted(hostnames, key=_slurm_hostname_key)
        except FileNotFoundError:
            hostnames = None
            logger.warning("scontrol not found, nodelist will be ignored. "
                           "You are probably running in the local mode.")

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
    # print(node2models)
    # for st, ed in node2models:
    #     if nodelist is not None and hostnames is not None:
    #         _this_nodelist = ",".join(hostnames[st:ed + 1])
    #     else:
    #         _this_nodelist = None
    #     node_count = ed - st + 1
    #     sched.model_worker.append(
    #         TasksGroup(
    #             count=node_count * device_mesh.n_gpus_per_node,
    #             scheduling=Scheduling.model_worker_default(
    #                 cpu=4,
    #                 gpu=1,
    #                 gpu_type="tesla",
    #                 mem=100000,
    #                 nodelist=_this_nodelist,
    #             ),
    #         ),)
    #     print(_this_nodelist)
    sched.model_worker.append(
        TasksGroup(
            count=device_mesh.n_nodes * device_mesh.n_gpus_per_node,
            scheduling=Scheduling.model_worker_default(
                cpu=4,
                gpu=1,
                gpu_type="tesla",
                mem=100000,
                nodelist=nodelist,
            ),
        ),)
    return sched


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
                cuda_cache_cleanliness=False,
                datasets=datasets,
                dataloader=dataloader,
                tokenizer_name_or_path=tokenizer_path,
            )
            for m in allocations.values():
                if m.mapping[i, j] and not any(m.rpc.model_name == s.id.model_name for s in mw.shards):
                    shard_idx = shard_counter[m.rpc.model_name]
                    if m.train_eval_config.optimizer.type != "empty":
                        backend = make_train_backend_config(m.train_eval_config)
                    else:
                        backend = make_inf_backend_config(m.train_eval_config)
                    mw.shards.append(
                        StandaloneModelShard(
                            id=ModelShardID(
                                model_name=m.rpc.model_name,
                                topo=m.topo,
                                dp_rank=m.topo.get_coord(shard_idx).data,
                                pp_rank=m.topo.get_coord(shard_idx).pipe,
                                mp_rank=m.topo.get_coord(shard_idx).model,
                            ),
                            model=model_configs[m.rpc.model_name],
                            backend=backend,
                        ))
                    shard_counter[m.rpc.model_name] += 1
            mw_configs.append(mw)
    return mw_configs


def auto_device_mapping(
    n_nodes: int,
    n_gpus_per_node: int = 8,
    mem: int = 80,
    nodelist: Optional[str] = None,
    mode: Literal["search", "model_pipe", "data_pipe", "full_model"] = "search",
):
    device_mesh = ClusterDeviceMesh(n_nodes, n_gpus_per_node, mem)

    def _auto_device_mapping(experiment_cls: Type,) -> Type[Experiment]:

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
                if mode == "search":
                    device_mapping_func = optimal_device_mapping
                elif mode == "model_pipe":
                    device_mapping_func = model_pipe_device_mapping
                elif mode == "data_pipe":
                    device_mapping_func = data_pipe_device_mapping
                elif mode == "test":
                    device_mapping_func = test_model_device_mapping
                elif mode == "full_model":
                    device_mapping_func = full_model_device_mapping
                else:
                    raise ValueError(f"Invalid mode {mode}")

                self._allocations = device_mapping_func(
                    device_mesh,
                    model_rpcs=self._internal_exp.rpcs,
                    model_configs=model_configs,
                    nodelist=nodelist,
                    num_gen_tokens=self._internal_exp.ppo.max_new_tokens,
                    n_ppo_minibatches=self._internal_exp.ppo.ppo_n_minibatches,
                )
                # import pprint
                # pprint.pprint(self._allocations)
                self._rpcs = [a.rpc for a in self._allocations.values()]

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
