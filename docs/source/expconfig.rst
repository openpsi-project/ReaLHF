Configurations
=============================

We illustrate configurations for quickstart experiments in this page.
Each type of experiment (e.g., SFT, PPO) corresponds to a specific 
configuration object (e.g., :class:`realhf.SFTConfig` for SFT).

Since ReaL uses `Hydra <https://hydra.cc/>`_ for configuration management,
users can override these options provided by the class recursively
with command line arguments.
Please check :doc:`quickstart` for concrete examples.

.. note::

    This page serves as a reference manual for the configuration objects,
    i.e., you can check which attributes can be modified and their default values.
    You don't need to read through this page before running experiments!

.. currentmodule:: realhf

Experiment Configurations
--------------------------

.. autoclass:: ExperimentSaveEvalControl

.. autoclass:: CommonExperimentConfig

.. autoclass:: SFTConfig

.. autoclass:: RWConfig

.. autoclass:: DPOConfig

.. autoclass:: GenerationHyperparameters

.. autoclass:: PPOHyperparameters

.. autoclass:: PPOConfig

.. autoclass:: GenerationConfig

Model Configurations
---------------------

.. autoclass:: ModelFamily

.. autoclass:: ModelTrainEvalConfig

.. autoclass:: OptimizerConfig

.. autoclass:: ParallelismConfig

.. autoclass:: AllocationConfig

.. autoclass:: ReaLModelConfig

Dataset Configurations
-----------------------

.. autoclass:: PromptAnswerDatasetConfig

.. autoclass:: PairedComparisonDatasetConfig

.. autoclass:: PromptOnlyDatasetConfig

``SequenceSample``
-----------------------

``SequenceSample`` is an object we use in model function calls.
It is inherited from the previous SRL project.

Users can regard it as a nested dictionary of arrays, except that indexing a ``SequenceSample`` results in *slicing every hosted arrays* (again, we don't use this feature in this project).

.. autoclass:: realhf.SequenceSample
    :members:

Dataflow Graph
-----------------

.. autoclass:: realhf.MFCDef