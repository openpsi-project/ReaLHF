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

Entrypoints to launch experiments. The major function to launch
experiments is ``main_start`` in ``main.py``. This function will set up
the scheduler, and the scheduler will run remote processes by ``python3
-m realhf.apps.remote``. In particular, the training command runs
``apps.main`` (probably on a CPU node), which will then start and wait
for all worker processes launched by ``apps.remote`` to finish.

-  `realhf/base/
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/base>`_

Some basic utilities imported and used by system or algorithm modules.

-  `realhf/experiments/
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/experiments>`_

Experiment configurations and registries. These objects are the ones
configured through Hydra commandline arguments, examplified in
:doc:`quickstart`.

-  `realhf/impl/
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl>`_

Implementation of datasets, models, model interfaces, and model
backends. Please refer to :doc:`impl` for the definition of these terms.

-  `realhf/scheduler
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/scheduler>`_

The scheduler used to launch and monitor experiments. It provides a
similar functionality to ``torchrun``, but is more flexible. The
scheduler is not related to any other ReaL's concepts or implementation
details such as parameter reallocation. It's just an infrastructure
utility.

-  `realhf/search_engine
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/search_engine>`_

The search engine used to find the best configuration for an experiment.
It does not work currently. 🙇🏻🙇🏻🙇🏻

-  `realhf/system
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/system>`_

The definition of workers, including the base worker, the model worker,
and the master worker.

-  `tests <https://github.com/openpsi-project/ReaLHF/tree/main/tests>`_

Unit tests for the codebase. Tests marked without ``pytest.mark.gpu`` or
``pytest.mark.distributed`` will be run upon each PR to main.

-  `examples
   <https://github.com/openpsi-project/ReaLHF/tree/main/examples>`_

Example scripts and customized experiments. Please see detailed
descriptions below.

-  `csrc <https://github.com/openpsi-project/ReaLHF/tree/main/csrc>`_

C++ and CUDA extensions, including CUDA kernels for computing GAE, CUDA
kernels used for reading and writing parameters in parameter
reallocation, and the C++ seach engine. The ``custom_all_reduce`` kernel
is adopted from vLLM but not used by ReaL. Kept here as legacy.

**********
 Examples
**********

Standalone Scripts
==================

-  `visualize_dfg.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/examples/visualize_dfg.py>`_

Used to visualize the dataflow graph given a list of
:class:`realhf.MFCDef` objects. Useful for understanding the algorithm
representation or when developing a new dataflow graph.

-  `load_and_eval_rw.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/examples/load_and_eval_rw.py>`_

The critic model created by ReaL cannot directly loaded by HuggingFace
or vLLM. This script serves as an example of loading the trained reward
model and evaluates it some validation data.

.. note::

   The checkpoints of actor models can be directly loaded by HuggingFace
   or vLLM.

Scripts for Running Existing Experiments
========================================

The corresponding subfolder shows examples using different scheduler
backends, either local subprocesses, SLURM, or Ray. They differ in the
``mode`` argument in commandline. Please check :doc:`quickstart` and
:doc:`distributed` for more details.

The meaning of the commandline arguments of SFT, reward modeling, DPO,
and PPO can be found in :doc:`quickstart` and :doc:`expconfig`. We make
complementary explanations about some special scripts below.

-  `local/gen.sh
   <https://github.com/openpsi-project/ReaLHF/tree/main/examples/scripts/local/gen.sh>`_

Loading trained checkpoints and running local generation using ReaL.
While the user may be more familar with HuggingFace's or vLLM's
generation pipeline, ReaL also provides a original generation pipeline
that achieves on-par throughput with vLLM. The benefit of using ReaL's
generation is that it can scale to larger models, multiple GPUs and
nodes easily. However, the generation options are limited compared with
vLLM (e.g., ReaL does not support beam-search). This script will run the
generate method in ``realhf/impl/model/interface/gen_interface.py``. The
user should modify this file to detokenize and save generated results
properly.

-  `local/ppo_manual.sh
   <https://github.com/openpsi-project/ReaLHF/tree/main/examples/scripts/local/ppo_manual.sh>`_
   and `local/ppo.sh
   <https://github.com/openpsi-project/ReaLHF/tree/main/examples/scripts/local/ppo.sh>`_

The former uses manual allocations for all MFCs while the latter uses a
hueristic allocation mode provided by ReaL. If the user wants to make a
manual allocation for distributed experiments, you can refer to
`local/ppo_manual.sh
<https://github.com/openpsi-project/ReaLHF/tree/main/examples/scripts/local/ppo_manual.sh>`_
to set device mesh strings properly.

Customized Algorithms
=====================

The customized algorithms basically involve implementing a new interface
file and a new experiment configuration file such that it can be
launched with the Hydra commandline API.

Beyond correct implementation, the salient point is to register the
interface and experiment configuration at the end of the script. Then,
the user can use ReaL's quickstart entrypoint to launch the customized
experiment just as the built-in algorithms. See the end of `grpo_exp.py
<https://github.com/openpsi-project/ReaLHF/tree/main/examples/new_algorithms/grpo/grpo_exp.py>`_
as an example.

The bash scripts provided in each algorithm folder are used to launch
the experiment.

-  `GRPO
   <https://github.com/openpsi-project/ReaLHF/tree/main/examples/new_algorithms/grpo/>`_
   (`GRPO paper <https://arxiv.org/pdf/2402.03300>`_)

GRPO differs from PPO in that it generates a group of answers for each
prompt, omits the critic model, and normalizes the advantage inside each
group. We repeat the prompts in the generation method of the interface,
and pack each group of answers as a single piece of data (see
:class:`realhf.SequenceSample` for details). We also change the
:class:`realhf.MFCDef` and the dataflow graph in the ``rpcs`` method in
the experiment configuration.

-  `ReMax
   <https://github.com/openpsi-project/ReaLHF/tree/main/examples/new_algorithms/reinforce/>`_
   (`ReMax paper <https://arxiv.org/abs/2310.10505>`_)

ReMax is a REINFORCE algorithm that uses the rewards of greedy
generations as the baseline. It omits the reference and the critic
model. The main changes compared with PPO are (1) the loss function is
changed from PPO to REINFORCE, (2) mini-batch updates are removed
because REINFORCE is purely on-policy, and (3) the dataflow graph is
changed to call generation and reward inference twice for sampled and
greedy generations respectively.

.. note::

   The changed dataflow graphs can be visualized by `visualize_dfg.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/examples/visualize_dfg.py>`_.

Customized Experiments
======================

-  `ppo_sentiment.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/examples/cuxtomized_exp/ppo_sentiment.py>`_

Implementing a new reward interface that uses a pre-trained HuggingFace
model to score the sentiment of generations, rather than a BT reward
model. It also changes PPO's configuration object to use the "null"
model and model backend for the reward inference MFC.

-  `ppo_ref_ema.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/examples/cuxtomized_exp/ppo_ref_ema.py>`_

PPO that gradually updates the reference model with exponential moving average,
   proposed in `this paper <https://arxiv.org/pdf/2404.10719>`_. It adds
   an uni-directional parameter reallocation hook from the MFC
   "actor_train" to "ref_inf". This hook will fire after the
   "actor_train" MFC is finished.

******
 APIs
******

Core APIs
=========

The core APIs directly configure workers and are in a lower level than
Quickstart APIs.

-  `config.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/core/config.py>`_

Basic concepts including :class:`realhf.ModelName`,
:class:`realhf.ModelShardID`, and :class:`realhf.ModelFamily`, and
objects served as worker-level configurations. The configuration objects
have similar attributes with a ``type_`` field and a ``args`` field. The
worker will find the registered class with the name ``type_`` and
initialize it with the ``args`` field. For example, see the
registeration of datasets under `realhf/impl/dataset/
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/dataset>`_
and the corresponding instantiation (i.e., ``data_api.make_dataset``) in
`realhf/system/model_worker.py
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/system/model_worker.py>`_.

-  `data_api.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/core/data_api.py>`_

It defines the basic data structure :class:`realhf.SequenceSample`, some
utility functions for data processing, and system-level configuration
APIs, such as ``register_dataset`` and ``make_dataset``.
``register_dataset`` will be called during importing dataset
implementations, and ``make_dataset`` will be called in workers given
the configuration objects defined in `config.py
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/core/config.py>`_.

-  `dfg.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/core/dfg.py>`_

Definition of the dataflow graph.

-  `model_api.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/core/model_api.py>`_

The system-level APIs of :class:`realhf.Model`,
:class:`realhf.ModelInterface`, and :class:`realhf.ModelBackend`, as
well as transformer configurations. It also defines the APIs to convert
a HuggingFace model to ``ReaLModel``.

-  `system_api.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/core/system_api.py>`_

The scheduling and worker-level configuration APIs.

HuggingFace Model Conversion
============================

We define model conversion utilities for various models under
`realhf/api/from_hf/
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/from_hf/>`_.
To add a new model, one must define functions to convert the model
config and state_dict back and forth. Please check :doc:`customization`
for detailed guides.

Quickstart APIs
===============

Quickstart APIs simplifies the worker-level configuration by exposing
several commonly used options to Hydra, e.g., the learning rate, batch
size, etc. These configurable objects have been documented in
:doc:`expconfig`.

-  `dataset.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/quickstart/dataset.py>`_

Configuration of three types of datasets.

-  `device_mesh.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/quickstart/device_mesh.py>`_

The definition of device mesh and MFC allocations.

-  `model.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/quickstart/model.py>`_

Configurations of the model, the optimizer, and the parallelism
strategy.

-  `entrypoint.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/quickstart/entrypoint.py>`_

The registry and entrypoint of quickstart experiments. Quickstart will
find the experiment runner from the Hydra configuration store.

-  `search.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/api/quickstart/search.py>`_

APIs for the ``search`` allocation mode. Currently not work. 🙇🏻🙇🏻🙇🏻

**************
 Base Modules
**************

Parallelism Strategy
====================

Suppose we have a cluster with shape (N, M), where N is the number of
nodes and M is the number of GPUs per node. ReaL will launch N * M model
worker processes, each exclusively occupying a GPU. These processes will
share a global PyTorch process group, and each MFC will create several
sub-groups on their own device meshes.

For example, suppose N=4, M=8, and we have MFC 1 occupying the first
half nodes, MFC 2 occupying the last three nodes, and MFC 3 occupying
the first node. ReaL will first create process groups on their device
meshes after creating the global group. Next, ReaL will create data,
tensor, and pipeline parallel groups inside each sub-group, similar to
Megatron-LM. These groups will kept in `constants.py
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/base/constants.py>`_
as per-process global constants.

In the above example, the first node is shared by MFC 1 and MFC 3. When
different MFCs are executed on the same GPU, ReaL switches the process
group by using a ``model_scope`` context defined in `constants.py
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/base/constants.py>`_.
The model name is given by the MFC. Under the scope, the 3D parallelism
groups specifially refers to the group of this MFC.

In summary, there are three level of process groups in ReaL. The first
level is the data/tensor/pipeline parallel group for a specific MFC. The
intermediate level is the "global" rank in the MFC's sub-group. The
outermost level is the global rank in the global group on all nodes. The
conversion from the first level to the second level is done by the
``ProcessTopology`` class in `topology.py
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/base/topology.py>`_,
and the conversion from the second level to the outermost level is done
by the ``rank_mapping`` dictionary in `constants.py
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/base/constants.py>`_.

-  `constants.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/base/constants.py>`_

It keeps the per-process global constants for the parallelism strategy,
environment variables, and default file paths for logging and
checkpoints.

-  `topology.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/base/topology.py>`_

The 3D topology of the device mesh, modified from DeepSpeed. The most
useful object in this file is ``ProcessTopology``, which can help
conviniently get the scalar sub-group rank given 3D parallelism ranks or
vice versa. ``PipeModelDataParallelTopology`` is kept for the
compatibility with DeepSpeed.

Data Packing
============

ReaL does not pad sequences by default. All sequences with variable
lengths are concatenated togather as a single 1D tensor and input to the
model, following the API of flash attention. While it's easy to split
padded sequences into mini-batches (i.e., slicing the batch dimension),
splitting concatenated sequences is not as straightforward. Partitioning
by the number of sequences can result in unbalanced mini-batches.

.. note::

   While packed sequences seem to reduce GPU memory, it enlarges memory
   fragmentation. We are unknown about the trade-off between the two.

To address this issue, ReaL implements a balanced partitioning algorithm
in `datapacking.py
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/base/datapacking.py>`_.
The algorithm ensures that the maximum length difference among differnt
mini-batches is minimized. It is called in the ``split`` method of
:class:`realhf.SequenceSample`.

There's also a ``reorder_to_balanced_batches`` function that reorders
the loaded data from the dataset such that (1) longer sequences appears
in earlier batches for detecting the OOM issue, and (2) each batch has a
nearly equal number of tokens. Since the optimal reordering is NP-hard,
the function uses a greedy algorithm to approximate the optimal
solution.

.. note::

   For PPO, the dataset reordering does not help to detect the OOM issue
   because the output length is determined by the Actor model.

Names and Name Resolving
========================

The launched workers require a synchronization mechanism to communicate
with each other. This is a achived by writing and reading values in a
distributed object store. All workers will wait until the required keys
are written before proceeding. It can be redis or simply a shared
directory on the filesystem. `names.py
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/base/names.py>`_
defines the keys to be written and `name_resolve.py
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/base/name_resolve.py>`_
defines the APIs to read and write values. When such an object store is
not available, there will be undefined behaviors.

Name resolving is used when setting up communication channels between
the master worker and model workers (i.e., when initializing the
request-reply stream, which is based on ZMQ sockets), and when
initializing pytorch process groups.

-  `names.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/base/names.py>`_

The keys to be written in the object store. Experiment and trial name
specific. There will be undefined behaviors if different trials use the
same name and are launched at the same time.

-  `gpu_utils.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/base/gpu_utils.py>`_

The utilities to isolate GPUs using name resolving. Each process will
reveal its own identity on the cluster, and will be assigned to an
unique GPU. ``CUDA_VISIBLE_DEVICES`` will be set accordingly and only a
single GPU should only be visible to this process.

-  `name_resolve.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/base/name_resolve.py>`_

The APIs to read and write values in the object store. The object store
can be redis or a shared directory on the filesystem.

***********************************
 Dataset and Model Implementations
***********************************

Datasets
========

In ReaL, datasets are created directly from JSON or JSONL files. There's
no prompt template in the code so the user must inject them manually.

Besides, ReaL requires that each piece of data has a unique entry called
"id", which is used to index data in the buffer of the master worker. It
can be any hashable objects, e.g., uuid, integer index, or a string.

For more details, please check the :doc:`customization` guide.

-  `prompt_answer_dataset.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/dataset/prompt_answer_dataset.py>`_

Used for SFT, where each piece of data should have a key "prompt" and
the corresponding "answer".

-  `prompt_dataset.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/dataset/prompt_dataset.py>`_

Used for PPO, where each piece of data should have a key "prompt".

-  `rw_paired_dataset.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/dataset/rw_paired_dataset.py>`_

Used for reward modeling, where each piece of data should have a key
"prompt", a list of "pos_answers", and a list of "neg_answers". Positive
and negative answers should be paired. Each prompt can have a different
number of comparison pairs.

Models
======

To optimize the performance of transformer models and support 3D
parallelism, we implement a class called ``ReaLModel`` in
`real_model_api.py
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/nn/real_model_api.py>`_.
It is more efficient than those implemented by HuggingFace transformers
and additionally supports parameter reallocation.

-  `nn
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/nn/>`_

The collection of codes implementing ``ReaLModel``. We also flatten the
parameters to support the parameter reallocation functionality.

-  `modules
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/modules/>`_

Neural network compoenents of ``ReaLModel``.

-  `parallelism
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/parallelism/>`_

Model/Tensor and pipeline parallelism support for ``ReaLModel``. Tensor
parallelism modules are copied and modified from Megatron-LM.

-  `conversion
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/conversion/>`_

Conversion utilities for HuggingFace models. Once registered,
``ReaLModel`` will have several new methods defined in
`realhf/impl/model/__init__.py
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/__init__.py>`_
For example, if a model family called "llama" is registered,
``ReaLModel`` will enable methods like "from_llama", "to_llama",
"config_from_llama", and "config_to_llama", which convert checkpoints
and configurations back and forth. If ``ReaLModel`` has loaded the HF
checkpoint with "from_llama", it will be able to save to the loaded
model family using "save_to_hf". ReaL enables distributed save and load
with 3D parallelism, and the saved checkpoint is fully compatible with
commonly used libraries like transformers and vLLM.

Model Backends
==============

All backends have the same signature :class:`realhf.ModelBacked` and
will produce an engine with signature :class:`realhf.PipelinableEngine`.
This engine will be used by all interface implementations.

-  `pipe_runner.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/backend/pipe_runner.py>`_

Implementing functions to run pipelined generate, inference, and
training instructions. It is not a backend, but will be used by the
model backends to run the model.

-  `inference.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/backend/inference.py>`_

The inference backend, which only supports inference and generation.

-  `deepspeed.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/backend/deepspeed.py>`_

The training backend based on DeepSpeed. Internally, it will also use
the inference engine for inference and generation. For training, it uses
the DeepSpeed ZeRO optimizer to train the model. That's the only
dependency on DeepSpeed.

.. note::

   The ZeRO optimizer in DeepSpeed will hold a copy of model parameters.
   Since the code of DS is quite messy, it's challenging to fetch it out
   and unify it with ReaL's parameter flattening mechanism. As a result,
   parameter reallocation with the DS backend will not correctly
   reallocate the *updated* parameters for other MFCs. The user may
   observe that the model is not learning at all.

   ReaL will raise errors if the user uses parameter reallocation with
   the DS backend.

-  `megatron.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/backend/megatron.py>`_

The training backend based on Megatron. Internally, it will also use the
inference engine for inference and generation. For training, it uses the
DDP module and the distributed optimizer from megatron. That's the only
dependency on megatron. Only the megatron backend supports parameter
reallocation.

Model Interfaces
================

An interface is a set of methods that will be called on a model. It is
algorithmically specific. An algorithm, represented as a dataflow graph
composed of MFCs, will subsequently find the configured interface
implementation in :class:`realhf.MFCDef`, and call the methods given by
the interface type (e.g., generate or inference).

Please check :class:`realhf.MFCDef` and :doc:`impl` for more details.

Communication
=============

The initialization of process groups and the communication algorithm for
data transfer and parameter reallocation are implemented under the `comm
<https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/comm>`_
folder. All communications are among GPUs based on NCCL.

-  `global_comm.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/comm/global_comm.py>`_

The initialization of the global process group and sub-groups for
different MFCs.

-  `data_transfer.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/comm/data_transfer.py>`_

The data transfer algorithm based on NCCL broadcast.

-  `param_realloc.py
   <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/comm/param_realloc.py>`_

The parameter reallocation algorithm based on NCCL broadcast.
