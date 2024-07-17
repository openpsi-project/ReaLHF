Configurations
=============================

.. note::

    This page serves as a reference manual for the configuration objects,
    i.e., you can check which attributes can be modified and their default values.
    You don't need to read through this page before running experiments!

    Please check the :doc:`quickstart` and :doc:`customization` sections for concrete examples
    of running experiments.

We illustrate configurations for quickstart experiments in this page.
Each type of experiment (e.g., SFT, PPO) corresponds to a specific 
configuration object (e.g., :class:`realhf.SFTConfig` for SFT).

Since ReaL uses `Hydra <https://hydra.cc/>`_ for configuration management,
users can override these options provided by the class recursively
with command line arguments.

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

Data Structure for Interfaces and Datasets
-----------------------------------------------

.. autoclass:: realhf.SequenceSample
    :members:

Dataflow Graph
-----------------

.. autoclass:: realhf.MFCDef