from collections import defaultdict
from statistics import mean, stdev
from typing import List, Optional, Union
import json
import os
import time

from flash_attn.bert_padding import unpad_input
import torch
import torch.distributed as dist
import transformers

from base.topology import *
from impl.model.nn.flash_mqat.flash_mqat_base import (FlashMQATBlock, FlashMQATConfig, OutputHead,
                                                      SequenceParallelActorHead, SequenceParallelCriticHead,
                                                      VocabPositionEmbedding)
from impl.model.utils.data import PipeCacheData, PipeTransferData
import api.config as config_package
import api.huggingface
import api.model
import base.constants


def make_layers(config: FlashMQATConfig, dtype, device):
    embedding_layer = VocabPositionEmbedding(
        config,
        dtype=dtype,
        device=device,
    )
    flash_mqat_blocks = [
        FlashMQATBlock(
            config,
            layer_index=i,
            output_layernorm=(i == 1),
            dtype=dtype,
            device=device,
        ) for i in range(2)
    ]

    # if config.is_critic and config.sequence_parallel:
    #     head = SequenceParallelCriticHead(
    #         config.hidden_dim,
    #         1,
    #         bias=False,
    #         device=device,
    #         dtype=dtype,
    #     )
    # elif not config.is_critic and base.constants.model_parallel_world_size() > 1:
    #     head = SequenceParallelActorHead(
    #         config.hidden_dim,
    #         config.vocab_size,
    #         bias=False,
    #         sequence_parallel=config.sequence_parallel,
    #         async_tensor_model_parallel_allreduce=not config.sequence_parallel,
    #         gradient_accumulation_fusion=config.gradient_accumulation_fusion,
    #         device=device,
    #         dtype=dtype,
    #     )
    # else:
    head = OutputHead(
        config.hidden_dim,
        1 if config.is_critic else config.vocab_size,
        bias=False,
        device=device,
        dtype=dtype,
    )

    layer_names = ["embedding_layer", "flash_mqat_block_0", "flash_mqat_block_1", "head"]
    return [embedding_layer] + flash_mqat_blocks + [head], layer_names


def make_stats_key(layer_name, name, bs, seq_len):
    return f"{layer_name}-{name}-{bs}-{seq_len}"


def parse_stats_key(key):
    # layer_name, name, bs, seq_len
    return key.split("-")


class ProfileLayers:

    def __init__(
        self,
        model_name: str,
        config: FlashMQATConfig,
        use_sequence_parallel: bool = False,
        use_gradient_checkpointing: bool = False,
        tokenizer: transformers.PreTrainedTokenizerFast = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model_name = model_name
        self.config = config
        self.backend_config = config_package.ModelBackend(
            type_="ds_train",
            args=dict(
                optimizer_name="adam",
                optimizer_config=dict(lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95)),
                warmup_steps_proportion=0.0,
                min_lr_ratio=0.0,
                zero_stage=1,
                engine_type="deepspeed",
                gradient_checkpointing=use_gradient_checkpointing,
                enable_fp16=True,
                enable_bf16=False,
                sequence_parallel=use_sequence_parallel,
                enable_async_p2p_communication=False,
            ),
        )

        self.dtype = dtype
        self.device = device
        self.layers, self.layer_names = make_layers(config, dtype, device)
        self.hidden_dim = config.hidden_dim
        self.head_dim = config.head_dim
        self.max_new_tokens = 128
        self.min_new_tokens = 128

        self.stats = defaultdict(list)
        self.num_layers = len(self.layers)

        self.layers = [
            api.model.Model(name, layer, tokenizer, device=device, dtype=dtype)
            for layer, name in zip(self.layers, self.layer_names)
        ]
        self.backend = api.model.make_backend(self.backend_config)
        ft_spec = api.model.FinetuneSpec(10, 100, 10, 32)
        self.layers = [self.backend.initialize(layer, ft_spec) for layer in self.layers]

    def reset_stats(self):
        self.stats = defaultdict(list)

    def sync_stats(self):
        for key, times in self.stats.items():
            times = torch.tensor(times, dtype=torch.float32, device=self.device)
            times_list = [torch.zeros_like(times) for _ in range(dist.get_world_size())]
            dist.all_gather(times_list, times)
            times = torch.cat(times_list)
            self.stats[key] = times.cpu().tolist()

    @torch.no_grad()
    def fwd_gen(self, bs, seq_len):
        input_ids = torch.randint(0,
                                  self.config.vocab_size, (bs, seq_len),
                                  dtype=torch.long,
                                  device=self.device)
        attention_mask = torch.ones_like(input_ids)
        # fwd_gen_0
        packed_input_ids, _, cu_seqlens, max_seqlen = unpad_input(input_ids, attention_mask)
        x = PipeTransferData(cu_seqlens=cu_seqlens, max_seqlen=int(max_seqlen), store_kv_cache=True)
        ys = [PipeCacheData() for _ in range(self.num_layers)]
        ys[0].input_ids = packed_input_ids

        for layer_name, layer, y in zip(self.layer_names, self.layers, ys):
            st = time.monotonic_ns()
            x = layer.module(x, y)
            x.pp_input = x.pp_output
            torch.cuda.synchronize()
            self.stats[make_stats_key(layer_name, "fwd_gen_0", bs, seq_len)].append(time.monotonic_ns() - st)

        prompt_logits = x.pp_output
        logits = prompt_logits[cu_seqlens[1:] - 1]
        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        cache_seqlens = input_lens.clone().to(dtype=torch.int32)
        layer_indices = range(len(ys))

        for y, layer_idx in zip(ys[1:-1], layer_indices[1:-1]):
            assert y.k_cache is not None and y.v_cache is not None and y.cache_seqlens is not None
            kvcache_seqlen = max(max_seqlen + self.max_new_tokens, self.hidden_dim // self.head_dim + 10)
            # fix of a flash attention bug
            k_cache = base.constants.get_global_memory_buffer().get_tensor(
                tensor_shape=(bs, kvcache_seqlen, *y.k_cache.shape[1:]),
                dtype=y.k_cache.dtype,
                name=f"kv_cache_{layer_idx}_k",
                force_zero=True)
            v_cache = base.constants.get_global_memory_buffer().get_tensor(
                tensor_shape=(bs, kvcache_seqlen, *y.v_cache.shape[1:]),
                dtype=y.v_cache.dtype,
                name=f"kv_cache_{layer_idx}_v",
                force_zero=True)
            indices = torch.arange(kvcache_seqlen, device=torch.cuda.current_device(),
                                   dtype=torch.long)[None, :] < input_lens[:, None]
            k_cache[indices] = y.k_cache
            v_cache[indices] = y.v_cache
            y.k_cache = k_cache
            y.v_cache = v_cache
            y.cache_seqlens = cache_seqlens
        x = PipeTransferData(store_kv_cache=True)
        ys[0].cache_seqlens = cache_seqlens

        # fwd_gen_1
        new_tokens = torch.randint(0, self.config.vocab_size, (bs, 1), dtype=torch.long, device=self.device)
        ys[0].input_ids = new_tokens
        ys[0].position_ids = None
        for layer_name, layer, y in zip(self.layer_names, self.layers, ys):
            st = time.monotonic_ns()
            x = layer.module(x, y)
            x.pp_input = x.pp_output
            torch.cuda.synchronize()
            self.stats[make_stats_key(layer_name, "fwd_gen_1", bs, seq_len)].append(time.monotonic_ns() - st)

    def fwd_bwd_opt(self, bs, seq_len):
        input_ids = torch.randint(0,
                                  self.config.vocab_size, (bs, seq_len),
                                  dtype=torch.long,
                                  device=self.device)
        attention_mask = torch.ones_like(input_ids)
        packed_input_ids, _, cu_seqlens, max_seqlen = unpad_input(input_ids, attention_mask)
        x = PipeTransferData(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, store_kv_cache=False)
        ys = [PipeCacheData() for _ in range(self.num_layers)]
        ys[0].input_ids = packed_input_ids

        for layer_name, layer, y in zip(self.layer_names, self.layers, ys):
            # fwd
            st = time.monotonic_ns()
            x = layer.module(x, y)
            torch.cuda.synchronize()
            self.stats[make_stats_key(layer_name, "fwd", bs, seq_len)].append(time.monotonic_ns() - st)
            # bwd
            r = torch.rand(*x.pp_output.shape, device=x.pp_output.device, dtype=x.pp_output.dtype)
            loss = torch.sum(x.pp_output * r)
            st = time.monotonic_ns()
            layer.module.backward(loss)
            torch.cuda.synchronize()
            self.stats[make_stats_key(layer_name, "bwd", bs, seq_len)].append(time.monotonic_ns() - st)
            # opt
            st = time.monotonic_ns()
            layer.module.step()
            torch.cuda.synchronize()
            self.stats[make_stats_key(layer_name, "opt", bs, seq_len)].append(time.monotonic_ns() - st)
            x.pp_input = x.pp_output.clone().detach()

    def print_stats(self):
        for key, times in self.stats.items():
            print(f"{key}: {mean(times)/1e3}, {stdev(times)/1e3}, "
                  f"{max(times)/1e3}, {min(times)/1e3}")

    def dump_stats(self, world_size, bs, seq_len):
        if dist.get_global_rank() == 0:
            # dump full stats
            dump_path = f"./profile_result/{self.model_name}/full-{world_size}-{bs}-{seq_len}.json"
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            with open(dump_path, "w") as f:
                json.dump(self.stats, f)

            # dump stats summary
            all_summary = {}
            for k, stats in self.stats.items():
                key_summary = dict(len=len(stats),
                                   mean=mean(stats) / 1e3,
                                   stdev=stdev(stats) / 1e3,
                                   max=max(stats) / 1e3,
                                   min=min(stats) / 1e3)
                all_summary[k] = key_summary
            dump_path = f"./profile_result/{self.model_name}/summary-{world_size}-{bs}-{seq_len}.json"
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            with open(dump_path, "w") as f:
                json.dump(all_summary, f)


def make_profile_layers(device: torch.device,
                        model_path: str,
                        model_name: str,
                        use_sequence_parallel: bool = False,
                        use_gradient_checkpointing: bool = False,
                        dtype: Optional[str] = None):
    if dtype == "fp16" or dtype == None:
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp32":
        dtype == torch.float32
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")
    tokenizer = None
    with open(os.path.join(model_path, "flash_mqat_config.json"), "r") as f:
        config = FlashMQATConfig(**json.load(f))
    config.sequence_parallel = use_sequence_parallel
    # m.load(model_path, init_critic_from_actor=False)
    if tokenizer is None:
        tokenizer = api.huggingface.load_hf_tokenizer(model_path)

    profile_layers = ProfileLayers(model_name,
                                   config,
                                   use_sequence_parallel=use_sequence_parallel,
                                   use_gradient_checkpointing=use_gradient_checkpointing,
                                   tokenizer=tokenizer,
                                   dtype=dtype,
                                   device=device)

    return profile_layers
