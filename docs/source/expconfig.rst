Configurations
=============================

We illustrate configurations for quickstart experiments in this page.
Each type of experiment (e.g., SFT, PPO) corresponds to a specific 
configuration class (e.g., :class:`realrlhf.SFTConfig` for SFT).

Since ReaL uses `Hydra <https://hydra.cc/>`_ for configuration management,
users can override these options provided by the class recursively
with command line arguments.
Please check :doc:`quickstart` for concrete examples.

.. currentmodule:: realrlhf

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

Dataset Configurations
-----------------------

.. autoclass:: PromptAnswerDatasetConfig

.. autoclass:: PairedComparisonDatasetConfig

.. autoclass:: PromptOnlyDatasetConfig
