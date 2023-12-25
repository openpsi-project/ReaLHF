# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from functools import partial
from typing import List, Optional, Tuple
import glob
import os
import re as regex

from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime import utils as ds_utils
from deepspeed.runtime.activation_checkpointing import checkpointing
from deepspeed.runtime.state_dict_factory import SDLoaderFactory
import torch
import torch.nn as nn

from base.datapack import partition_balanced as true_partition_balanced
from base.monitor import process_memory_mb, time_mark
from base.topology import PipeDataParallelTopology, PipelineParallelGrid, PipeModelDataParallelTopology
from impl.model.nn.flash_mqat.flash_mqat_base import FlashMQATConfig, OutputHead
from impl.model.utils.data import PipeCacheData, PipeTransferData
from impl.model.utils.save_load import load_from_disk, save_to_disk
import base.constants
import base.logging as logging

logger = logging.getLogger("Pipeline Module")


class PipelineError(Exception):
    """Errors related to the use of deepspeed.PipelineModule """


class LayerSpec:
    """Building block for specifying pipeline-parallel modules.

    LayerSpec stores the type information and parameters for each stage in a
    PipelineModule. For example:

    .. code-block:: python

        nn.Sequence(
            torch.nn.Linear(self.in_dim, self.hidden_dim, bias=False),
            torch.nn.Linear(self.hidden_hidden, self.out_dim)
        )

    becomes

    .. code-block:: python

        layer_specs = [
            LayerSpec(torch.nn.Linear, self.in_dim, self.hidden_dim, bias=False),
            LayerSpec(torch.nn.Linear, self.hidden_hidden, self.out_dim)]
        ]
    """

    def __init__(self, typename, *module_args, **module_kwargs):
        self.typename = typename
        self.module_args = module_args
        self.module_kwargs = module_kwargs

        if not issubclass(typename, nn.Module):
            raise RuntimeError('LayerSpec only supports torch.nn.Module types.')

        if dist.is_initialized():
            self.global_rank = base.constants.parallelism_rank()
        else:
            self.global_rank = -1

    def __repr__(self):
        return ds_utils.call_to_str(self.typename.__name__, self.module_args, self.module_kwargs)

    def build(self, log=False):
        """Build the stored specification."""
        if log:
            logger.info(f'RANK={self.global_rank} building {repr(self)}')

        return self.typename(*self.module_args, **self.module_kwargs)


class TiedLayerSpec(LayerSpec):

    def __init__(self,
                 key,
                 typename,
                 *module_args,
                 forward_fn=None,
                 tied_weight_attr='weight',
                 **module_kwargs):
        super().__init__(typename, *module_args, **module_kwargs)
        self.key = key
        self.forward_fn = forward_fn
        self.tied_weight_attr = tied_weight_attr


class PipelineModule(nn.Module):
    ### This module should only be used by pipeline engine!!
    """Modules to be parallelized with pipeline parallelism.

    The key constraint that enables pipeline parallelism is the
    representation of the forward pass as a sequence of layers
    and the enforcement of a simple interface between them. The
    forward pass is implicitly defined by the module ``layers``. The key
    assumption is that the output of each layer can be directly fed as
    input to the next, like a ``torch.nn.Sequence``. The forward pass is
    implicitly:

    .. code-block:: python

        def forward(self, inputs):
            x = inputs
            for layer in self.layers:
                x = layer(x)
            return x

    .. note::
        Pipeline parallelism is not compatible with ZeRO-2 and ZeRO-3.

    Args:
        layers (Iterable): A sequence of layers defining pipeline structure. Can be a ``torch.nn.Sequential`` module.
        num_stages (int, optional): The degree of pipeline parallelism. If not specified, ``topology`` must be provided.
        topology (``deepspeed.runtime.pipe.ProcessTopology``, optional): Defines the axes of parallelism axes for training. Must be provided if ``num_stages`` is ``None``.
        loss_fn (callable, optional): Loss is computed ``loss = loss_fn(outputs, label)``
        seed_layers(bool, optional): Use a different seed for each layer. Defaults to False.
        seed_fn(type, optional): The custom seed generating function. Defaults to random seed generator.
        base_seed (int, optional): The starting seed. Defaults to 1234.
        partition_method (str, optional): The method upon which the layers are partitioned. Defaults to 'parameters'.
        # do not support activation checkpoint now
        activation_checkpoint_interval (int, optional): The granularity activation checkpointing in terms of number of layers. 0 disables activation checkpointing.
        activation_checkpoint_func (callable, optional): The function to use for activation checkpointing. Defaults to ``deepspeed.checkpointing.checkpoint``.
        checkpointable_layers(list, optional): Checkpointable layers may not be checkpointed. Defaults to None which does not additional filtering.
    """

    def __init__(self,
                 layers,
                 num_stages=None,
                 topology=None,
                 loss_fn=None,
                 seed_layers=False,
                 seed_fn=None,
                 base_seed=1234,
                 is_critic=False,
                 partition_method='parameters',
                 config: Optional[FlashMQATConfig] = None,
                 dtype=None,
                 device=None,
                 activation_checkpoint_interval=0,
                 activation_checkpoint_func=checkpointing.checkpoint,
                 checkpointable_layers=None):

        super().__init__()

        if num_stages is None and topology is None:
            raise RuntimeError('must provide num_stages or topology')

        self.is_critic = is_critic
        self.dtype = dtype
        self.device = device
        self.config = config  # get underlying config

        self.micro_offset = 0

        self.loss_fn = loss_fn

        self.seed_layers = seed_layers
        self.seed_fn = seed_fn
        self.base_seed = base_seed
        if base.constants.parallelism_rank() == 0:
            try:
                seed_str = self.seed_fn.__name__
            except AttributeError:
                seed_str = None
            print(f'SEED_LAYERS={self.seed_layers} BASE_SEED={self.base_seed} SEED_FN={seed_str}')

        # Setup world info
        self.world_group = base.constants.parallelism_group()
        logger.info(f'Ranks in world group: {torch.distributed.get_process_group_ranks(self.world_group)}')
        self.global_rank = dist.get_rank(group=self.world_group)
        self.world_size = dist.get_world_size(group=self.world_group)
        self.local_rank = int(os.environ.get("LOCAL_RANK", None))
        logger.info(("GPU rank info:", self.world_group, self.global_rank, self.world_size, self.local_rank))
        assert self.local_rank != None

        if topology:
            self._topo = topology
            self.num_stages = self._topo.get_dim('pipe')
        else:
            self.num_stages = num_stages
            if topology is None:
                if self.world_size % self.num_stages != 0:
                    raise RuntimeError(
                        f'num_stages ({self.num_stages}) must divide distributed world size ({self.world_size})'
                    )
                dp = self.world_size // num_stages
                topology = PipeModelDataParallelTopology(num_pp=num_stages, num_mp=1, num_dp=dp)
                self._topo = topology

        # Construct communicators for pipeline topology
        self._grid = base.constants.grid()

        self.stage_id = self._topo.get_coord(self.global_rank).pipe
        print(f"rank {torch.distributed.get_rank()} pipeline stage ID: {self.stage_id}")

        # Initialize partition information
        self._layer_specs = list(layers)
        self._num_layers = len(self._layer_specs)
        self._local_start = 0
        self._local_stop = None
        self._partition_layers(method=partition_method)

        self.forward_funcs = []
        self.fwd_map = {}
        self.tied_modules = nn.ModuleDict()
        self.tied_weight_attrs = {}

        # Offset the random seed by the stage ID.
        #newseed = get_accelerator().initial_seed() + self._grid.get_stage_id()
        #ds_utils.set_random_seed(newseed)

        #with torch.random.fork_rng(devices=[get_accelerator().current_device_name()]):
        self._build()
        self.to(get_accelerator().device_name(self.local_rank))

        self.tied_comms = self._index_tied_modules()
        self._synchronize_tied_weights()

        # for saving checkpoints
        self.num_checkpoint_shards = 1

    @property
    def get_config(self):
        return self.config

    def _build(self):
        specs = self._layer_specs

        for local_idx, layer in enumerate(specs[self._local_start:self._local_stop]):
            layer_idx = local_idx + self._local_start
            if self.seed_layers:
                if self.seed_fn:
                    self.seed_fn(self.base_seed + layer_idx)
                else:
                    ds_utils.set_random_seed(self.base_seed + layer_idx)

            # Recursively build PipelineModule objects
            if isinstance(layer, PipelineModule):
                raise NotImplementedError('RECURSIVE BUILD NOT YET IMPLEMENTED')

            # LayerSpec objects contain an nn.Module that should be allocated now.
            elif isinstance(layer, nn.Module):
                name = str(layer_idx)
                self.forward_funcs.append(layer)
                self.fwd_map.update({name: len(self.forward_funcs) - 1})
                self.add_module(name, layer)

            # TiedLayerSpec objects contain an nn.Module that should be allocated now.
            elif isinstance(layer, TiedLayerSpec):
                # Build and register the module if we haven't seen it before.
                if layer.key not in self.tied_modules:
                    self.tied_modules[layer.key] = layer.build()
                    self.tied_weight_attrs[layer.key] = layer.tied_weight_attr

                if layer.forward_fn is None:
                    # Just use forward()
                    self.forward_funcs.append(self.tied_modules[layer.key])
                else:
                    # User specified fn with args (module, input)
                    self.forward_funcs.append(partial(layer.forward_fn, self.tied_modules[layer.key]))

            # LayerSpec objects contain an nn.Module that should be allocated now.
            elif isinstance(layer, LayerSpec):
                # substitute output head if self.is_critic=True
                # TODO: hack, find better solution
                if layer_idx == self._num_layers - 1 and self.is_critic:
                    layer = LayerSpec(
                        layer.typename,
                        self.config.hidden_dim,
                        1,
                        bias=False,
                        device=self.device,
                        dtype=self.dtype,
                    )
                module = layer.build()
                name = str(layer_idx)
                self.forward_funcs.append(module)
                self.fwd_map.update({name: len(self.forward_funcs) - 1})
                self.add_module(name, module)

            # Last option: layer may be a functional (e.g., lambda). We do nothing in
            # that case and just use it in forward()
            else:
                self.forward_funcs.append(layer)

        # All pipeline parameters should be considered as model parallel in the context
        # of our FP16 optimizer
        for p in self.parameters():
            p.ds_pipe_replicated = False

    def _count_layer_params(self):
        """Count the trainable parameters in individual layers.

        This routine will only build one layer at a time.

        Returns:
            A list of the number of parameters in each layer.
        """
        param_counts = [0] * len(self._layer_specs)
        for idx, layer in enumerate(self._layer_specs):
            if isinstance(layer, LayerSpec):
                l = layer.build()
                params = filter(lambda p: p.requires_grad, l.parameters())
                param_counts[idx] = sum(p.numel() for p in params)
            elif isinstance(layer, nn.Module):
                params = filter(lambda p: p.requires_grad, layer.parameters())
                param_counts[idx] = sum(p.numel() for p in params)
        return param_counts

    def _find_layer_type(self, layername):
        idxs = []
        typeregex = regex.compile(layername, regex.IGNORECASE)
        for idx, layer in enumerate(self._layer_specs):
            name = None
            if isinstance(layer, LayerSpec):
                name = layer.typename.__name__
            elif isinstance(layer, nn.Module):
                name = layer.__class__.__name__
            else:
                try:
                    name = layer.__name__
                except AttributeError:
                    continue
            if typeregex.search(name):
                idxs.append(idx)

        if len(idxs) == 0:
            raise RuntimeError(f"Partitioning '{layername}' found no valid layers to partition.")
        return idxs

    @property
    def num_layers(self):
        """ number of layers in current pipeline stage
        """
        return len(self.forward_funcs)

    def gradient_checkpointing_enable(self, attn=False, mlp=False):
        for layer in self.forward_funcs:
            try:
                layer.gradient_checkpointing_enable(attn, mlp)
            except AttributeError:
                logger.warning(f'Layer {layer} does not support gradient checkpointing, skipping ...')

    def forward(self, x: PipeTransferData, ys: List[PipeCacheData]):
        local_micro_offset = self.micro_offset + 1

        for idx, (layer, y) in enumerate(zip(self.forward_funcs, ys)):
            time_mark(f"layer_{idx}_start", dist.get_rank(group=self.world_group))
            self.curr_layer = idx + self._local_start
            if self.seed_layers:
                new_seed = (self.base_seed * local_micro_offset) + self.curr_layer
                if self.seed_fn:
                    self.seed_fn(new_seed)
                else:
                    ds_utils.set_random_seed(new_seed)

            x = layer(x, y)
            x.pp_input = x.pp_output
            x.pp_output = None
            time_mark(f"layer_{idx}_end", dist.get_rank(group=self.world_group))

        return x, ys

    def _partition_layers(self, method='parameters'):
        num_stages = self._topo.get_dim('pipe')
        stage_id = self._topo.get_coord(self.global_rank).pipe

        if self.global_rank == 0:
            logger.info(f'Partitioning pipeline stages with method {method}')

        method = method.lower()

        if method == 'uniform':
            num_layers = len(self._layer_specs)
            self.parts = ds_utils.partition_uniform(num_items=num_layers, num_parts=num_stages)
        elif method == 'parameters':
            logger.warning(
                "method 'parameters' does not partition parameters to each stage evenly, "
                "preserving the option as default due to checkpoints already partitioned in this way."
                "Update to use 'parameters_balanced' option instead!"
                "Change the option both in model partition script and pipe/pipe+model model wrapper configs!")
            param_counts = self._count_layer_params()
            self.parts = ds_utils.partition_balanced(weights=param_counts, num_parts=num_stages)
        elif method == 'parameters_balanced':
            param_counts = self._count_layer_params()
            import numpy as np
            param_counts = np.array(param_counts)
            self.parts = true_partition_balanced(nums=param_counts, k=num_stages)
        elif method.startswith('type:'):
            layertype = method.split(':')[1]
            binary_weights = [0] * len(self._layer_specs)
            for idx in self._find_layer_type(layertype):
                binary_weights[idx] = 1
            self.parts = ds_utils.partition_balanced(weights=binary_weights, num_parts=num_stages)
        elif method == 'profile':
            raise NotImplementedError(f'Partitioning method {method} not implemented.')
        else:
            raise NotImplementedError(f'Partitioning method {method} not implemented.')

        # Print some information on the partitioning.
        if self.global_rank == 0:
            for stage in range(num_stages):
                start = self.parts[stage]
                stop = self.parts[stage + 1]
                print(f'stage={stage} layers={stop - start}')
                for idx, layer in enumerate(self._layer_specs[start:stop]):
                    name = str(layer)
                    if isinstance(layer, LayerSpec):
                        name = layer.typename.__name__
                    if isinstance(layer, nn.Module):
                        name = layer.__class__.__name__
                    else:
                        try:
                            name = layer.__name__
                        except AttributeError:
                            pass
                    print(f'    {idx+start:2d}: {name}')
            if self.loss_fn:
                try:
                    print(f'  loss: {self.loss_fn.__name__}')
                except AttributeError:
                    print(f'  loss: {self.loss_fn.__class__.__name__}')

        self._set_bounds(start=self.parts[stage_id], stop=self.parts[stage_id + 1])

    def allreduce_tied_weight_gradients(self):
        '''All reduce the gradients of the tied weights between tied stages'''
        for key, comm in self.tied_comms.items():
            weight = getattr(self.tied_modules[key], comm['weight_attr'])
            dist.all_reduce(weight.grad, group=comm['group'])

    def get_tied_weights_and_groups(self):
        weight_group_list = []
        for key, comm in self.tied_comms.items():
            weight = getattr(self.tied_modules[key], comm['weight_attr'])
            weight_group_list.append((weight, comm['group']))
        return weight_group_list

    def _synchronize_tied_weights(self):
        for key, comm in self.tied_comms.items():
            dist.broadcast(
                getattr(comm['module'], comm['weight_attr']),
                src=min(comm['ranks']) + base.constants.process_group_offset(),
                group=comm['group'],
            )

    def _index_tied_modules(self):
        ''' Build communication structures for tied modules. '''
        tied_comms = {}
        if self._topo.get_dim('pipe') == 1:
            return tied_comms

        specs = self._layer_specs
        tie_keys = set(s.key for s in specs if isinstance(s, TiedLayerSpec))
        for key in tie_keys:
            # Find the layers that the tied module appears in
            tied_layers = []
            for idx, layer in enumerate(specs):
                if isinstance(layer, TiedLayerSpec) and layer.key == key:
                    tied_layers.append(idx)
            # Find all stages with this tied module
            # TODO: Would be nice to remove the nested data/model parallelism loops and
            # TODO: instead generalize in some way, since we really just care about the
            # TODO: stage that owns the tied layer. Then loop over each (dp, mp, ...)
            # TODO: fiber to generate process groups.
            tied_stages = set(self.stage_owner(idx) for idx in tied_layers)
            for dp in range(self._grid.data_parallel_size):
                for mp in range(self._grid.get_slice_parallel_world_size()):
                    tied_ranks = []
                    for s in sorted(tied_stages):
                        if self._grid.get_slice_parallel_world_size() > 1:
                            tied_ranks.append(self._grid.stage_to_global(stage_id=s, data=dp, model=mp))
                        else:
                            tied_ranks.append(self._grid.stage_to_global(stage_id=s, data=dp))
                    group = dist.new_group(
                        ranks=[rank + base.constants.process_group_offset() for rank in tied_ranks])

                    # Record this tied module if we own a local copy of it.
                    if self.global_rank in tied_ranks:
                        assert key in self.tied_modules
                        if key in self.tied_modules:
                            tied_comms[key] = {
                                'ranks': tied_ranks,
                                'group': group,
                                'weight_attr': self.tied_weight_attrs[key],
                                'module': self.tied_modules[key],
                            }
                            # Only count the tied module once in the eyes of the FP16 optimizer
                            if self.global_rank != tied_ranks[0]:
                                for p in self.tied_modules[key].parameters():
                                    p.ds_pipe_replicated = True
        if len(tied_comms) > 0:
            raise NotImplementedError(
                "Tied weights require dist.new_group of a specific model, which are not supported!")
        return tied_comms

    def partitions(self):
        return self.parts

    def stage_owner(self, layer_idx):
        assert 0 <= layer_idx < self._num_layers
        for stage in range(self._topo.get_dim('pipe')):
            if self.parts[stage] <= layer_idx < self.parts[stage + 1]:
                return stage
        raise RuntimeError(f'Layer {layer_idx} not owned? parts={self.parts}')

    def _set_bounds(self, start=None, stop=None):
        """Manually define the range of layers that will be built on this process.

        These boundaries are treated as list slices and so start is inclusive and stop is
        exclusive. The default of None for both results in all layers being built
        locally.
        """
        self._local_start = start
        self._local_stop = stop

    def set_checkpoint_interval(self, interval):
        assert interval >= 0
        self.checkpoint_interval = interval

    def topology(self):
        """ ProcessTopology object to query process mappings. """
        return self._topo

    def mpu(self):
        return self._grid

    def num_pipeline_stages(self):
        return self._topo.get_dim('pipe')

    def ckpt_prefix(self, checkpoints_path, tag):
        """Build a prefix for all checkpoint files written by this module. """
        # All checkpoint files start with this
        rank_name = 'module'

        # Data parallelism is omitted from the naming convention because we are agnostic
        # to this in the checkpoint.
        omit_dims = frozenset(['data'])
        axes = [a for a in self._grid._topo.get_axis_names() if a not in omit_dims]
        for dim in axes:
            rank = getattr(self._grid._topo.get_coord(rank=self.global_rank), dim)
            rank_name += f'-{dim}_{rank:02d}'

        ckpt_name = os.path.join(checkpoints_path, str(tag), rank_name)
        return ckpt_name

    def ckpt_layer_path(self, ckpt_dir, local_layer_idx):
        """Customize a prefix for a specific pipeline module layer. """
        idx = local_layer_idx + self._local_start
        layer_ckpt_path = os.path.join(ckpt_dir, f'layer_{idx:02d}')
        rank_repr = self._grid._topo.get_rank_repr(rank=self.global_rank)
        if rank_repr != '':
            layer_ckpt_path += f'-{rank_repr}'
        layer_ckpt_path += '-model_states.pt'
        return layer_ckpt_path

    def ckpt_layer_path_list(self, ckpt_dir, local_layer_idx):
        """Get all ckpt file list for a specific pipeline module layer. """
        idx = local_layer_idx + self._local_start
        layer_ckpt_path = os.path.join(ckpt_dir, f'layer_{idx:02d}-')
        layer_ckpt_path += "*model_states.pt"
        ckpt_files = glob.glob(layer_ckpt_path)
        ckpt_files.sort()
        return ckpt_files

    def save(self, save_dir):
        dp_rank = self._grid.data_parallel_id
        if dp_rank > 0:  # only save on dp_rank = 0
            return

        tp_rank = self._grid.get_tensor_model_parallel_rank()
        save_to_disk(self.state_dict(),
                     save_dir,
                     output_fn=f"pytorch_model-pp-{self.stage_id:02d}-mp-{tp_rank:02d}-s-" +
                     "{shard:02d}.bin",
                     save_type="pt",
                     n_shards=self.num_checkpoint_shards)

    def load(self, load_dir, init_critic_from_actor: bool = False):
        state_dict, n_shards = load_from_disk(
            load_dir,
            fn_pattern=r".*" +
            f"-pp-{self.stage_id:02d}-mp-{self._grid.get_tensor_model_parallel_rank():02d}-" + r"s-(\d{2}).*",
            return_n_shards=True)

        if init_critic_from_actor and f'{self.config.n_layers + 1}.weight' in state_dict:
            state_dict.pop(f'{self.config.n_layers + 1}.weight')
            self.load_state_dict(state_dict, strict=False)
        else:
            self.load_state_dict(state_dict)

        self.num_checkpoint_shards = n_shards
        logger.info("Loaded model from {}".format(load_dir))
        process_memory_mb("after_load_state_dict")
        self._synchronize_tied_weights()

    def _is_checkpointable(self, funcs):
        # This is an unfortunate hack related to torch and deepspeed activation checkpoint implementations.
        # Some layers like torch.nn.Embedding will not receive grads if checkpointed, which breaks things.
        # I presume it's related to the discrete inputs that cannot require_grad? Need to revisit.
        if self.__class__.__name__ in ('GPTModelPipe', 'GPT2ModelPipe'):
            return all('ParallelTransformerLayerPipe' in f.__class__.__name__ for f in funcs)
        if self.checkpointable_layers is not None:
            return all(f.__class__.__name__ in self.checkpointable_layers for f in funcs)

        params = [f.parameters() for f in funcs if isinstance(f, torch.nn.Module)]
        return any(len(list(p)) > 0 for p in params)
