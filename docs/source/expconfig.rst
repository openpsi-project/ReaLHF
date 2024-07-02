Configurations
=============================

We illustrate configurations for quickstart experiments in this page.
Each type of experiment (e.g., SFT, PPO) corresponds to a specific 
configuration object (e.g., :class:`realhf.SFTConfig` for SFT).

Since ReaL uses `Hydra <https://hydra.cc/>`_ for configuration management,
users can override these options provided by the class recursively
with command line arguments.
Please check :doc:`quickstart` for concrete examples.

.. currentmodule:: realhf

Experiment Configurations
--------------------------

.. autoclass:: CommonExperimentConfig

.. autoclass:: SFTConfig

.. autoclass:: RWConfig

.. autoclass:: DPOConfig

.. autoclass:: PPOHyperparameters

.. autoclass:: PPOConfig

Model Configurations
---------------------

.. autoclass:: ModelTrainEvalConfig

.. autoclass:: OptimizerConfig

.. autoclass:: ParallelismConfig

.. autoclass:: AllocationConfig

.. autoclass:: realhf.ReaLModelConfig

.. autoclass:: realhf.impl.model.nn.real_llm_api.ReaLModel
    :members:
    :undoc-members:
    :exclude-members: forward, state_dict, load_state_dict, build_reparallelization_plan, build_reparallelized_layers_async, patch_reparallelization, pre_process, post_process, shared_embedding_or_output_weight

Dataset Configurations
-----------------------

.. autoclass:: PromptAnswerDatasetConfig

.. autoclass:: PairedComparisonDatasetConfig

.. autoclass:: PromptOnlyDatasetConfig

``NamedArray``
-----------------------

``NamedArray`` is an object we use in model function calls.
It is inherited from the previous SRL project.

Named array extends plain arrays/tensors in the following ways.

1. NamedArray aggregates multiple arrays, possibly of different shapes.
2. Each array is given a name, providing a user-friendly way of indexing to the corresponding data.
3. NamedArrays can be nested. (Although it should *not* be nested in this system.)
4. NamedArray can store metadata such as sequence lengths, which is useful for padding and masking without causing CUDA synchronization.

Users can regard it as a nested dictionary of arrays, except that indexing a ``NamedArray`` results in *slicing every hosted arrays* (again, we don't use this feature in this project).

.. autoclass:: realhf.base.namedarray.NamedArray
    :members:

.. autofunction::realhf.base.namedarray.from_dict

.. autofunction::realhf.base.namedarray.recursive_aggregate

.. autofunction::realhf.base.namedarray.recursive_apply

Dataflow Graph
-----------------

.. autoclass:: realhf.MFCDef