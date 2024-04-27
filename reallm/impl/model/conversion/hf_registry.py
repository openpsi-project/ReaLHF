from typing import *
import dataclasses
import json
import os
import time

import torch
import torch.distributed as dist
import transformers

from reallm.api.core import model_api
from reallm.base import constants, logging
from reallm.base.saveload_utils import split_state_dict_into_shards
from reallm.impl.model.nn.real_llm_api import ReaLModel
from reallm.impl.model.nn.real_llm_parallel import mp_merge_key, mp_partition_real_model_state_dict

logger = logging.getLogger("HF Registry")


@dataclasses.dataclass
class HFModelRegistry:
    name: str
    hf_cls_name: str
    config_from_hf_converter: Callable[[transformers.PretrainedConfig], model_api.ReaLModelConfig]
    config_to_hf_converter: Callable[[model_api.ReaLModelConfig], transformers.PretrainedConfig]
    sd_from_hf_converter: Callable[[Dict, model_api.ReaLModelConfig], Dict]
    sd_to_hf_converter: Callable[[Dict, model_api.ReaLModelConfig], Dict]
    embedding_param_names: Callable[[model_api.ReaLModelConfig], List[str]]
    tblock_param_names: Callable[[model_api.ReaLModelConfig, int], List[str]]
    head_param_names: Callable[[model_api.ReaLModelConfig], List[str]]

    def config_from_hf(
        self,
        hf_config: Optional[transformers.PretrainedConfig] = None,
        model_path: Optional[str] = None,
        is_critic: bool = False,
    ) -> model_api.ReaLModelConfig:
        if hf_config is None:
            hf_config = transformers.AutoConfig.from_pretrained(os.path.join(model_path, "config.json"))
        config = self.config_from_hf_converter(hf_config)
        config.is_critic = is_critic
        return config

    def config_to_hf(self, real_config: model_api.ReaLModelConfig) -> transformers.PretrainedConfig:
        return self.config_to_hf_converter(real_config)

    def load(self, model: ReaLModel, load_dir: str, init_critic_from_actor: bool = False):
        # NOTE: moving this upwards will result in circular import

        tik = time.perf_counter()
        with open(os.path.join(load_dir, "config.json"), "r") as f:
            hf_config = json.load(f)
        assert (self.hf_cls_name == hf_config["architectures"][0]
                ), f"{self.hf_cls_name} != {hf_config['architectures'][0]}"

        layer_indices = range(model.layer_idx_start, model.layer_idx_end)

        required_hf_sd_names = []
        for lidx in layer_indices:
            if lidx == 0:
                required_hf_sd_names += self.embedding_param_names(model.config)
            elif lidx == model.config.n_layers + 1:
                required_hf_sd_names += self.head_param_names(model.config)
            else:
                required_hf_sd_names += self.tblock_param_names(model.config, lidx - 1)

        if os.path.exists(os.path.join(load_dir, "pytorch_model.bin.index.json")):
            with open(os.path.join(load_dir, "pytorch_model.bin.index.json"), "r") as f:
                hf_sd_mapping = json.load(f)["weight_map"]
            files_to_load = set(hf_sd_mapping[name] for name in required_hf_sd_names)
        else:
            files_to_load = ["pytorch_model.bin"]
        setup_time = time.perf_counter() - tik

        load_times, partition_times = [], []
        state_dict = {}
        for fn in files_to_load:
            load_tik = time.perf_counter()
            # set map_location to be CPU is a little bit faster
            sd = torch.load(os.path.join(load_dir, fn), map_location="cpu")
            partition_tik = time.perf_counter()
            sd = {k: v for k, v in sd.items() if k in required_hf_sd_names}
            sd = self.sd_from_hf_converter(sd, model.config)
            state_dict.update(
                mp_partition_real_model_state_dict(
                    sd,
                    model.config,
                    constants.model_parallel_world_size(),
                    constants.model_parallel_rank(),
                ))
            load_times.append(partition_tik - load_tik)
            partition_times.append(time.perf_counter() - partition_tik)

        copy_tik = time.perf_counter()
        if init_critic_from_actor and f"{model.config.n_layers + 1}.weight" in state_dict:
            state_dict.pop(f"{model.config.n_layers + 1}.weight")
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=True)

        # Some logging info
        copy_time = time.perf_counter() - copy_tik
        load_times = "[" + ", ".join(f"{t:.2f}" for t in load_times) + "]"
        partition_times = "[" + ", ".join(f"{t:.2f}" for t in partition_times) + "]"
        if os.getenv("REAL_LOG_LOAD_TIME", None) == "1":
            logger.info(
                f"Loading from HuggingFace Model setup time cost={setup_time:.2f}s, load time cost={load_times}, "
                f"partition time cost={partition_times}, copy time cost={copy_time:.2f}s")

    def save(
        self,
        model: ReaLModel,
        tokenizer: transformers.PreTrainedTokenizer,
        save_dir: str,
        epoch: Optional[int] = None,
        epoch_step: Optional[int] = None,
        global_step: Optional[int] = None,
    ):
        dp_rank = constants.data_parallel_rank()
        if dp_rank > 0:  # only save on dp_rank = 0
            return

        subfolder = ""
        if epoch is not None:
            subfolder += f"epoch{epoch}"
        if epoch_step is not None:
            subfolder += f"epochstep{epoch_step}"
        if global_step is not None:
            subfolder += f"globalstep{global_step}"
        save_dir = os.path.join(save_dir, subfolder)
        os.makedirs(save_dir, exist_ok=True)

        pp_rank = constants.pipe_parallel_rank()
        mp_rank = constants.model_parallel_rank()
        mp_size = constants.model_parallel_world_size()
        pp_size = constants.pipe_parallel_world_size()

        # We will gather parameters across the model parallel group,
        # and save parameters to separate shards across the pipeline parallel group.

        # To decrease the size of each saved file, we also split the file
        # of each pipeline stage into smaller shards.
        approx_param_size = sum(v.numel() * v.element_size() for v in model.state_dict().values()) * mp_size
        max_shard_size_byte = os.getenv("REAL_SAVE_MAX_SHARD_SIZE_BYTE", int(1e10))
        n_shards_this_stage = (approx_param_size + max_shard_size_byte - 1) // max_shard_size_byte
        n_shards_this_stage = torch.tensor(n_shards_this_stage, dtype=torch.int32, device="cuda")
        pp_stage_n_shards = [torch.zeros(1, dtype=torch.int32, device="cuda") for _ in range(pp_size)]
        dist.all_gather(pp_stage_n_shards, n_shards_this_stage, group=constants.pipe_parallel_group())
        pp_stage_n_shards = [int(n.item()) for n in pp_stage_n_shards]
        assert all(x >= 1 for x in pp_stage_n_shards)

        # Gather parameters across the model parallel group.
        sd = model.state_dict()
        cpu_sd = {}
        for k, v in sd.items():
            # For simplicity, we gather all parameters no matter whether
            # it is indeed partitioned.
            gather_list = [torch.zeros_like(v) for _ in range(mp_size)]
            dist.all_gather(gather_list, v, group=constants.model_parallel_group())
            gathered = mp_merge_key(k, gather_list, model.config)
            cpu_sd[k] = gathered.cpu()

        if mp_rank > 0:
            return

        hf_sd = self.sd_to_hf_converter(cpu_sd, model.config)
        hf_config = self.config_to_hf_converter(model.config)

        param_size = sum([value.numel() * value.element_size() for value in hf_sd.values()])
        param_size = torch.tensor(param_size, dtype=torch.int64, device="cuda")
        dist.all_reduce(param_size, op=dist.ReduceOp.SUM, group=constants.pipe_parallel_group())
        param_size = param_size.item()

        if pp_rank == 0:
            hf_config.to_json_file(os.path.join(save_dir, "config.json"))
            tokenizer.save_vocabulary(save_dir)

        # Dump parameters to disk.
        if len(pp_stage_n_shards) == 1 and pp_stage_n_shards[0] == 1:
            fn = "pytorch_model.bin"
            torch.save(hf_sd, os.path.join(save_dir, fn))
        else:
            output_fn = "pytorch_model" + "-{shard:05d}" + f"-of-{sum(pp_stage_n_shards):05d}.bin"

            n_shards = pp_stage_n_shards[pp_rank]
            shard_offset = sum(pp_stage_n_shards[:pp_rank])

            shards = split_state_dict_into_shards(hf_sd, n_shards)

            bin_index = {}
            bin_index["metadata"] = dict(total_size=param_size)
            bin_index["weight_map"] = {}
            for i, shard in enumerate(shards):
                shard_idx = shard_offset + i
                torch.save(shard, os.path.join(save_dir, output_fn.format(shard=shard_idx + 1)))
                for k in shard:
                    bin_index["weight_map"][k] = output_fn.format(shard=shard_idx + 1)

            if pp_rank == 0:
                with open(os.path.join(save_dir, "pytorch_model.bin.index.json"), "w") as f:
                    json.dump(bin_index, f, indent=4)
