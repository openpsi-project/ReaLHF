###################
 Code Architecture
###################

**********
 Overview
**********

-  `realhf/api/
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api>`_

Definitions of the dataflow graph, data and model APIs, configuration
objects, and the implementation of transforming HuggingFace models to
ReaL.

-  `realhf/apps/
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/apps>`_

Entry points for launching experiments. The primary function to launch
experiments is ``main_start`` in `main.py
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/apps/main.py>`_.
This function sets up the scheduler, which runs remote processes via
``python3 -m realhf.apps.remote``. Specifically, the training command
runs ``apps.main`` (likely on a CPU node), which then starts and waits
for all worker processes launched by `remote.py
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/apps/remote.py>`_
to finish.

-  `realhf/base/
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/base>`_

Basic utilities imported and used by system or algorithm modules.

-  `realhf/experiments/
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/experiments>`_

Experiment configurations and registries. These objects are configured
through Hydra command-line arguments, as exemplified in the
:doc:`quickstart`.

-  `realhf/impl/
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl>`_

Implementation of datasets, models, model interfaces, and model
backends. Please refer to :doc:`impl` for definitions of these terms.

-  `realhf/scheduler
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/scheduler>`_

The scheduler used to launch and monitor experiments. It provides
functionality similar to ``torchrun``, but with greater flexibility. The
scheduler is independent of other ReaL concepts or implementation
details, such as parameter reallocation. It's purely an infrastructure
utility.

-  `realhf/search_engine
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/search_engine>`_

The search engine used to find the best configuration for an experiment.
It is currently non-functional. 🙇🏻🙇🏻🙇🏻

-  `realhf/system
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/system>`_

Definitions of workers, including the base worker, the model worker, and
the master worker.

-  `tests <https://github.com/openpsi-project/ReaLHF/tree/main/tests>`_

Unit tests for the codebase. Tests not marked with ``pytest.mark.gpu``
or ``pytest.mark.distributed`` will run upon each PR to the main branch.

-  `examples
   <https://github.com/openpsi-project/ReaLHF/tree/main/examples>`_

Example scripts and customized experiments. Please see the detailed
descriptions below.

-  `csrc <https://github.com/openpsi-project/ReaLHF/tree/main/csrc>`_

C++ and CUDA extensions, including CUDA kernels for computing GAE, CUDA
kernels used for reading and writing parameters during parameter
reallocation, and the C++ search engine. The ``custom_all_reduce``
kernel is adopted from vLLM but is not used by ReaL. It is kept here as
legacy.

**********
 Examples
**********

Standalone Scripts
==================

-  `visualize_dfg.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/examples/visualize_dfg.py>`_

This script is used to visualize the dataflow graph given a list of
:class:`realhf.MFCDef` objects. It is useful for understanding the
algorithm's representation or when developing a new dataflow graph.

-  `load_and_eval_rw.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/examples/load_and_eval_rw.py>`_

The critic model created by ReaL cannot be directly loaded by
HuggingFace or vLLM. This script provides an example of how to load the
trained reward model and evaluate it on some validation data.

.. note::

   The checkpoints of actor models can be directly loaded by HuggingFace
   or vLLM.

Scripts for Running Existing Experiments
========================================

The corresponding subfolder contains examples using different scheduler
backends, including local subprocesses, SLURM, or Ray. They differ in
the ``mode`` argument in the command line. Please refer to
:doc:`quickstart` and :doc:`distributed` for more details.

The meanings of the command-line arguments for SFT, reward modeling,
DPO, and PPO can be found in :doc:`quickstart` and :doc:`expconfig`.
Below, we provide additional explanations for some specific scripts.

-  `local/gen.sh
   <https://github.com/openpsi-project/ReaLHF/tree/main/examples/scripts/local/gen.sh>`_

This script loads trained checkpoints and runs local generation using
ReaL. While users may be more familiar with HuggingFace's or vLLM's
generation pipelines, ReaL also offers an original generation pipeline
that achieves throughput on par with vLLM. The advantage of using ReaL's
generation pipeline is its scalability to larger models, multiple GPUs,
and nodes. However, the generation options are more limited compared to
vLLM (e.g., ReaL does not support beam search). This script runs the
generate method in ``realhf/impl/model/interface/gen_interface.py``.
Users should modify this file to properly detokenize and save the
generated results.

-  `local/ppo_manual.sh
   <https://github.com/openpsi-project/ReaLHF/tree/main/examples/scripts/local/ppo_manual.sh>`_
   and `local/ppo.sh
   <https://github.com/openpsi-project/ReaLHF/tree/main/examples/scripts/local/ppo.sh>`_

The former script uses manual allocations for all MFCs, while the latter
uses a heuristic allocation mode provided by ReaL. If users wish to make
manual allocations for distributed experiments, they can refer to
`local/ppo_manual.sh
<https://github.com/openpsi-project/ReaLHF/tree/main/examples/scripts/local/ppo_manual.sh>`_
to properly set device mesh strings.

Customized algorithms typically involve implementing a new interface
file and a new experiment configuration file so that the experiment can
be launched using the Hydra command-line API.

Beyond correct implementation, it is crucial to register the interface
and experiment configuration at the end of the script. This allows users
to launch the customized experiment through ReaL's quickstart entry
point, just like the built-in algorithms. For an example, see the end of
`grpo_exp.py
<https://github.com/openpsi-project/ReaLHF/tree/main/examples/new_algorithms/grpo/grpo_exp.py>`_.

The bash scripts provided in each algorithm folder are used to launch
the experiments.

-  `GRPO
   <https://github.com/openpsi-project/ReaLHF/tree/main/examples/new_algorithms/grpo/>`_
   (`GRPO paper <https://arxiv.org/pdf/2402.03300>`_)

GRPO differs from PPO in that it generates a group of answers for each
prompt, omits the critic model, and normalizes the advantage within each
group. We repeat the prompts in the generation method of the interface
and package each group of answers as a single piece of data (see
:class:`realhf.SequenceSample` for details). We also modify the
:class:`realhf.MFCDef` and the dataflow graph in the ``rpcs`` method of
the experiment configuration.

-  `ReMax
   <https://github.com/openpsi-project/ReaLHF/tree/main/examples/new_algorithms/reinforce/>`_
   (`ReMax paper <https://arxiv.org/abs/2310.10505>`_)

ReMax is a REINFORCE algorithm that uses the rewards of greedy
generations as the baseline. It omits the reference and the critic
model. The main changes compared to PPO are: (1) the loss function is
modified from PPO to REINFORCE, (2) mini-batch updates are removed
because REINFORCE is purely on-policy, and (3) the dataflow graph is
altered to call generation and reward inference twice for sampled and
greedy generations, respectively.

.. note::

   The modified dataflow graphs can be visualized using
   `visualize_dfg.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/examples/visualize_dfg.py>`_.

Customized Experiments
======================

-  `ppo_sentiment.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/examples/cuxtomized_exp/ppo_sentiment.py>`_

Implements a new reward interface that uses a pre-trained HuggingFace
model to score the sentiment of generated content, instead of a BT
reward model. It also modifies PPO's configuration object to use the
"null" model and model backend for reward inference in the MFC.

-  `ppo_ref_ema.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/examples/cuxtomized_exp/ppo_ref_ema.py>`_

PPO with gradual updates to the reference model using an exponential moving average,
   as proposed in `this paper <https://arxiv.org/pdf/2404.10719>`_. It
   adds a unidirectional parameter reallocation hook from the MFC
   "actor_train" to "ref_inf". This hook is triggered after the
   "actor_train" MFC is completed.

******
 APIs
******

Core APIs
=========

The core APIs directly configure workers and operate at a lower level
than the Quickstart APIs.

-  `config.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/core/config.py>`_

Basic concepts include :class:`realhf.ModelName`,
:class:`realhf.ModelShardID`, and :class:`realhf.ModelFamily`, which
serve as worker-level configurations. The configuration objects
typically have a ``type_`` field and an ``args`` field. The worker will
locate the registered class using the name in ``type_`` and initialize
it with the ``args`` field. For example, see the registration of
datasets under `realhf/impl/dataset/
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/dataset>`_
and the corresponding instantiation (e.g., ``data_api.make_dataset``) in
`realhf/system/model_worker.py
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/system/model_worker.py>`_.

-  `data_api.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/core/data_api.py>`_

Defines the basic data structure :class:`realhf.SequenceSample`, along
with utility functions for data processing and system-level
configuration APIs, such as ``register_dataset`` and ``make_dataset``.
``register_dataset`` is called when importing dataset implementations,
and ``make_dataset`` is invoked by workers using the configuration
objects defined in `config.py
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/core/config.py>`_.

-  `dfg.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/core/dfg.py>`_

Defines the dataflow graph.

-  `model_api.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/core/model_api.py>`_

Provides the system-level APIs for :class:`realhf.Model`,
:class:`realhf.ModelInterface`, and :class:`realhf.ModelBackend`, along
with transformer configurations. It also defines the APIs to convert a
HuggingFace model into a ``ReaLModel``.

-  `system_api.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/core/system_api.py>`_

Defines the scheduling and worker-level configuration APIs.

HuggingFace Model Conversion
============================

Model conversion utilities for various models are defined under
`realhf/api/from_hf/
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/from_hf/>`_.
To add a new model, you must define functions to convert the model's
config and state_dict back and forth. Please refer to
:doc:`customization` for detailed guides.

Quickstart APIs
===============

Quickstart APIs simplify worker-level configuration by exposing several
commonly used options to Hydra, such as learning rate, batch size, etc.
These configurable objects are documented in :doc:`expconfig`.

-  `dataset.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/quickstart/dataset.py>`_

Configuration for three types of datasets.

-  `device_mesh.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/quickstart/device_mesh.py>`_

Defines the device mesh and MFC allocations.

-  `model.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/quickstart/model.py>`_

Configurations for the model, optimizer, and parallelism strategy.

-  `entrypoint.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/quickstart/entrypoint.py>`_

The registry and entrypoint for quickstart experiments. Quickstart will
locate the experiment runner from the Hydra configuration store.

-  `search.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/quickstart/search.py>`_

APIs for the ``search`` allocation mode. Currently not functional.
🙇🏻🙇🏻🙇🏻

**************
 Base Modules
**************

Parallelism Strategy
====================

-  `constants.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/base/constants.py>`_

   Contains the per-process global constants for the parallelism
   strategy, environment variables, and default file paths for logging
   and checkpoints.

-  `topology.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/base/topology.py>`_

   Defines the 3D topology of the device mesh, modified from DeepSpeed.
   The most useful object in this file is ``ProcessTopology``, which can
   conveniently obtain the scalar sub-group rank given 3D parallelism
   ranks, or vice versa. ``PipeModelDataParallelTopology`` is retained
   for compatibility with DeepSpeed.

Data Packing
============

ReaL does not pad sequences by default. All sequences with variable
lengths are concatenated together into a single 1D tensor and input to
the model, following the API of flash attention. While it's easy to
split padded sequences into mini-batches (i.e., slicing the batch
dimension), splitting concatenated sequences is not as straightforward.
Partitioning by the number of sequences can result in unbalanced
mini-batches.

.. note::

   While packed sequences may reduce GPU memory usage, they can increase
   memory fragmentation. We are unsure about the trade-off between the
   two.

To address this issue, ReaL implements a balanced partitioning algorithm
in `datapacking.py
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/base/datapacking.py>`_.
The algorithm ensures that the maximum length difference among different
mini-batches is minimized. It is invoked in the ``split`` method of
:class:`realhf.SequenceSample`.

There is also a ``reorder_to_balanced_batches`` function that reorders
the loaded data from the dataset such that (1) longer sequences appear
in earlier batches to detect out-of-memory (OOM) issues, and (2) each
batch has a nearly equal number of tokens. Since the optimal reordering
is NP-hard, the function uses a greedy algorithm to approximate the
optimal solution.

.. note::

   For PPO, dataset reordering does not help detect OOM issues because
   the output length is determined by the Actor model.

Names and Name Resolving
========================

The launched workers require a synchronization mechanism to communicate
with each other. This is achieved by writing and reading values in a
distributed object store. All workers will wait until the required keys
are written before proceeding. This can be implemented using Redis or
simply a shared directory on the filesystem. `names.py
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/base/names.py>`_
defines the keys to be written, and `name_resolve.py
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/base/name_resolve.py>`_
defines the APIs to read and write values. When such an object store is
not available, undefined behavior may occur.

Name resolving is used when setting up communication channels between
the master worker and model workers (e.g., when initializing the
request-reply stream based on ZMQ sockets) and when initializing PyTorch
process groups.

-  `names.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/base/names.py>`_

   Defines the keys to be written in the object store. These keys are
   specific to experiments and trials. Undefined behavior may occur if
   different trials use the same name and are launched simultaneously.

-  `gpu_utils.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/base/gpu_utils.py>`_

   Provides utilities for isolating GPUs using name resolving. Each
   process will reveal its own identity on the cluster and will be
   assigned a unique GPU. The ``CUDA_VISIBLE_DEVICES`` environment
   variable will be set accordingly, ensuring that only a single GPU is
   visible to this process.

-  `name_resolve.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/base/name_resolve.py>`_

   Defines the APIs for reading and writing values in the object store.
   The object store can be Redis or a shared directory on the
   filesystem.

***********************************
 Dataset and Model Implementations
***********************************

Datasets
========

In ReaL, datasets are created directly from JSON or JSONL files. There
is no prompt template in the code, so users must inject them manually.

Additionally, ReaL requires that each piece of data has a unique entry
called "id," which is used to index data in the buffer of the master
worker. This "id" can be any hashable object, such as a UUID, integer
index, or string.

For more details, please refer to the :doc:`customization` guide.

-  `prompt_answer_dataset.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/dataset/prompt_answer_dataset.py>`_

   Used for SFT, where each piece of data should have a key "prompt" and
   the corresponding "answer."

-  `prompt_dataset.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/dataset/prompt_dataset.py>`_

   Used for PPO, where each piece of data should have a key "prompt."

-  `rw_paired_dataset.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/dataset/rw_paired_dataset.py>`_

   Used for reward modeling, where each piece of data should have a key
   "prompt," a list of "pos_answers," and a list of "neg_answers."
   Positive and negative answers should be paired. Each prompt can have
   a different number of comparison pairs.

Models
======

To optimize the performance of transformer models and support 3D
parallelism, we have implemented a class called ``ReaLModel`` in
`real_model_api.py
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/nn/real_model_api.py>`_.
It is more efficient than the implementations provided by HuggingFace
Transformers and additionally supports parameter reallocation.

-  `nn
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/nn/>`_

   Contains the code for implementing ``ReaLModel``. This module also
   flattens the parameters to support the parameter reallocation
   functionality.

-  `modules
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/modules/>`_

   Includes the neural network components of ``ReaLModel``.

-  `parallelism
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/parallelism/>`_

   Provides model/tensor and pipeline parallelism support for
   ``ReaLModel``. The tensor parallelism modules are copied and modified
   from Megatron-LM.

-  `conversion
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/conversion/>`_

   Contains utilities for converting HuggingFace models. Once
   registered, ``ReaLModel`` will have several new methods defined in
   `realhf/impl/model/__init__.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/__init__.py>`_.
   For example, if a model family called "llama" is registered,
   ``ReaLModel`` will support methods like "from_llama", "to_llama",
   "config_from_llama", and "config_to_llama", which convert checkpoints
   and configurations back and forth. If ``ReaLModel`` has loaded an HF
   checkpoint using "from_llama", it will be able to save to the loaded
   model family using "save_to_hf". ReaL enables distributed saving and
   loading with 3D parallelism, and the saved checkpoint is fully
   compatible with commonly used libraries such as Transformers and
   vLLM.

Model Backends
==============

All backends share the same signature, :class:`realhf.ModelBackend`, and
produce an engine with the signature :class:`realhf.PipelinableEngine`.
This engine will be used by all interface implementations.

-  `pipe_runner.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/backend/pipe_runner.py>`_

   Implements functions to run pipelined generation, inference, and
   training instructions. It is not a backend itself but is used by the
   model backends to execute the model.

-  `inference.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/backend/inference.py>`_

   The inference backend, which supports only inference and generation.

-  `deepspeed.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/backend/deepspeed.py>`_

   The training backend based on DeepSpeed. Internally, it also uses the
   inference engine for inference and generation. For training, it
   employs the DeepSpeed ZeRO optimizer. This is the only dependency on
   DeepSpeed.

   .. note::

      The ZeRO optimizer in DeepSpeed holds a copy of the model
      parameters. Due to the complexity of the DeepSpeed code,
      integrating it with ReaL's parameter flattening mechanism is
      challenging. As a result, parameter reallocation with the
      DeepSpeed backend may not correctly reallocate the *updated*
      parameters for other MFCs. Users might observe that the model does
      not learn as expected.

      ReaL will raise errors if parameter reallocation is attempted with
      the DeepSpeed backend.

-  `megatron.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/backend/megatron.py>`_

   The training backend based on Megatron. Internally, it also uses the
   inference engine for inference and generation. For training, it uses
   the DDP module and the distributed optimizer from Megatron. This is
   the only dependency on Megatron. Only the Megatron backend supports
   parameter reallocation.

Model Interfaces
================

An interface is a set of methods that will be called on a model. It is
algorithmically specific. An algorithm, represented as a dataflow graph
composed of MFCs, will subsequently find the configured interface
implementation in :class:`realhf.MFCDef` and call the methods defined by
the interface type (e.g., generate or inference).

Please refer to :class:`realhf.MFCDef` and :doc:`impl` for more details.

Communication
=============

The initialization of process groups and the communication algorithm for
data transfer and parameter reallocation are implemented in the `comm
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/comm>`_
folder. All communications between GPUs are based on NCCL.

-  `global_comm.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/comm/global_comm.py>`_

   Initializes the global process group and sub-groups for different
   MFCs.

-  `data_transfer.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/comm/data_transfer.py>`_

   Implements the data transfer algorithm based on NCCL broadcast.

-  `param_realloc.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/comm/param_realloc.py>`_

   Implements the parameter reallocation algorithm based on NCCL
   broadcast.
