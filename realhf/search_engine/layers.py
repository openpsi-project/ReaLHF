import os
import pickle
import time
from collections import defaultdict
from typing import List, Optional, Union

import pandas as pd
import torch
import torch.distributed as dist
import transformers

import realhf.api.core.model_api as model_api
import realhf.api.core.system_api as config_package
import realhf.base.constants as constants
import realhf.base.logging as logging
from realhf.api.core.model_api import ReaLModelConfig
from realhf.impl.model.utils.padding import unpad_input

logger = logging.getLogger("profile layers", "system")


def make_layers(config: ReaLModelConfig, dtype, device):
    from realhf.impl.model.nn.real_llm_base import (
        OutputHead,
        ReaLModelBlock,
        VocabPositionEmbedding,
    )

    embedding_layer = VocabPositionEmbedding(
        config,
        dtype=dtype,
        device=device,
    )
    real_model_blocks = [
        ReaLModelBlock(
            config,
            layer_index=i,
            output_layernorm=(i == 1),
            dtype=dtype,
            device=device,
        )
        for i in range(1)
    ]
    head = OutputHead(
        config.hidden_dim,
        1 if config.is_critic else config.vocab_size,
        bias=False,
        device=device,
        dtype=dtype,
    )

    layer_names = ["embedding_layer", "block_0", "head"]
    return [embedding_layer] + real_model_blocks + [head], layer_names


class ProfileLayers:

    def __init__(
        self,
        model_name: str,
        config: ReaLModelConfig,
        tokenizer: transformers.PreTrainedTokenizerFast = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model_name = model_name
        self.config = config
        self.backend_config = config_package.ModelBackend(
            type_="deepspeed",
            args=dict(
                optimizer_name="adam",
                optimizer_config=dict(lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95)),
                warmup_steps_proportion=0.0,
                min_lr_ratio=0.0,
                zero_stage=1,
                enable_fp16=True,
                enable_bf16=False,
            ),
        )

        self.dtype = dtype
        self.device = device
        self.layers, self.layer_names = make_layers(config, dtype, device)
        self.hidden_dim = config.hidden_dim
        self.head_dim = config.head_dim
        self.max_new_tokens = 128  # only useful in kv cache memory alloc
        self.min_new_tokens = 128

        self.stats = defaultdict(list)
        self.num_layers = len(self.layers)

        self.layers = [
            model_api.Model(name, layer, tokenizer, device=device, dtype=dtype)
            for layer, name in zip(self.layers, self.layer_names)
        ]
        self.backend = model_api.make_backend(self.backend_config)
        ft_spec = model_api.FinetuneSpec(10, 100, 10)
        self.layers = [self.backend.initialize(layer, ft_spec) for layer in self.layers]
        self.stats = defaultdict(list)

    def reset_stats(self):
        self.stats = defaultdict(list)

    def insert_data_point(self, layer_name, name, bs, seq_len, time_ns):
        self.stats["layer_name"].append(layer_name)
        self.stats["op_name"].append(name)
        self.stats["bs"].append(bs)
        self.stats["seq_len"].append(seq_len)
        self.stats["time_ns"].append(time_ns)

    @torch.no_grad()
    def fwd_gen(self, bs, seq_len):
        from realhf.impl.model.nn.real_llm_base import PipeCacheData, PipeTransferData

        input_ids = torch.randint(
            0,
            self.config.vocab_size,
            (bs, seq_len),
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.ones_like(input_ids, device=self.device)
        # fwd_gen_0
        packed_input_ids, _, cu_seqlens, max_seqlen = unpad_input(
            input_ids, attention_mask
        )
        cu_seqlens = cu_seqlens.to(device=self.device)
        packed_input_ids = packed_input_ids.to(device=self.device)
        x = PipeTransferData(
            cu_seqlens=cu_seqlens,
            max_seqlen=int(max_seqlen),
            store_kv_cache=True,
        )
        ys = [PipeCacheData() for _ in range(self.num_layers)]
        ys[0].packed_input_ids = packed_input_ids

        for layer_name, layer, y in zip(self.layer_names, self.layers, ys):
            st = time.monotonic_ns()
            x: PipeTransferData = layer.module(x, y)
            x.pp_input = x.pp_output
            torch.cuda.synchronize()
            self.insert_data_point(
                layer_name, "fwd_gen_0", bs, seq_len, time.monotonic_ns() - st
            )

        prompt_logits = x.pp_output
        logits = prompt_logits[cu_seqlens[1:] - 1]
        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        cache_seqlens = input_lens.clone().to(dtype=torch.int32)
        layer_indices = range(len(ys))

        for y, layer_idx in zip(ys[1:-1], layer_indices[1:-1]):
            assert (
                y.k_cache is not None
                and y.v_cache is not None
                and y.cache_seqlens is not None
            )
            kvcache_seqlen = max(
                max_seqlen + self.max_new_tokens,
                self.hidden_dim // self.head_dim + 10,
            )
            # fix of a flash attention bug
            k_cache = torch.zeros(
                (bs, kvcache_seqlen, *y.k_cache.shape[1:]),
                dtype=y.k_cache.dtype,
                device=self.device,
            )
            v_cache = torch.zeros_like(k_cache)
            indices = (
                torch.arange(
                    kvcache_seqlen,
                    device=torch.cuda.current_device(),
                    dtype=torch.long,
                )[None, :]
                < input_lens[:, None]
            )
            k_cache[indices] = y.k_cache
            v_cache[indices] = y.v_cache
            y.k_cache = k_cache
            y.v_cache = v_cache
            y.cache_seqlens = cache_seqlens
        x = PipeTransferData(store_kv_cache=True)
        ys[0].cache_seqlens = cache_seqlens

        # fwd_gen_1
        new_tokens = torch.randint(
            0,
            self.config.vocab_size,
            (bs,),
            dtype=torch.long,
            device=self.device,
        )
        ys[0].packed_input_ids = new_tokens
        ys[0].packed_position_ids = None
        x.cu_seqlens = torch.arange(bs + 1, dtype=torch.int32, device=self.device)
        x.max_seqlen = 1
        for layer_name, layer, y in zip(self.layer_names, self.layers, ys):
            st = time.monotonic_ns()
            x = layer.module(x, y)
            x.pp_input = x.pp_output
            torch.cuda.synchronize()
            self.insert_data_point(
                layer_name, "fwd_gen_1", bs, seq_len, time.monotonic_ns() - st
            )

    def fwd_bwd_opt(self, bs, seq_len):
        from realhf.impl.model.nn.real_llm_base import PipeCacheData, PipeTransferData

        input_ids = torch.randint(
            0,
            self.config.vocab_size,
            (bs, seq_len),
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.ones_like(input_ids, device=self.device)
        packed_input_ids, _, cu_seqlens, max_seqlen = unpad_input(
            input_ids, attention_mask
        )
        cu_seqlens = cu_seqlens.to(device=self.device)
        packed_input_ids = packed_input_ids.to(device=self.device)
        x = PipeTransferData(
            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, store_kv_cache=False
        )
        ys = [PipeCacheData() for _ in range(self.num_layers)]
        ys[0].packed_input_ids = packed_input_ids

        for layer_name, layer, y in zip(self.layer_names, self.layers, ys):
            # fwd
            st = time.monotonic_ns()
            x: PipeTransferData = layer.module(x, y)
            torch.cuda.synchronize()
            self.insert_data_point(
                layer_name, "fwd", bs, seq_len, time.monotonic_ns() - st
            )
            # bwd
            r = torch.rand(
                *x.pp_output.shape,
                device=x.pp_output.device,
                dtype=x.pp_output.dtype,
            )
            loss = torch.max(x.pp_output * r)
            st = time.monotonic_ns()
            layer.module.backward(loss)
            torch.cuda.synchronize()
            self.insert_data_point(
                layer_name, "bwd", bs, seq_len, time.monotonic_ns() - st
            )
            # opt
            st = time.monotonic_ns()
            layer.module.step()
            torch.cuda.synchronize()
            self.insert_data_point(
                layer_name, "opt", bs, seq_len, time.monotonic_ns() - st
            )
            x.pp_input = x.pp_output.clone().detach()

    def make_dataframe_and_print(self):
        df = pd.DataFrame(self.stats)
        logger.info(f"Current Stats: \nstr{df}")

    def dump_stats(self, world_size):
        rank = dist.get_rank()
        # dump full stats
        dump_dir = os.path.join(
            constants.PROFILER_CACHE_PATH,
            "layer_stats",
        )
        dump_path = os.path.join(
            dump_dir, self.model_name, f"layer-stats_{world_size}_{rank}.pkl"
        )
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)

        with open(dump_path, "wb") as f:
            df = pd.DataFrame(self.stats)
            pickle.dump(df, f)


def make_profile_layers(
    device: torch.device,
    model_path: str,
    model_name: str,
    dtype: Optional[str] = None,
    hf_model_type: str = "llama",
):
    from realhf.impl.model.nn.real_llm_api import ReaLModel

    if dtype == "fp16" or dtype == None:
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp32":
        dtype == torch.float32
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")
    tokenizer = None
    config: ReaLModelConfig = getattr(ReaLModel, f"config_from_{hf_model_type}")(
        model_path=model_path,
    )
    if tokenizer is None:
        tokenizer = model_api.load_hf_tokenizer(model_path)

    profile_layers = ProfileLayers(
        model_name, config, tokenizer=tokenizer, dtype=dtype, device=device
    )

    return profile_layers
