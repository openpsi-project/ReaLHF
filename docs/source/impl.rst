##############################
 Implementation Details
##############################

**************************
 Algorithm Representation
**************************

An algorithm in ReaL are represented as a *dataflow graph* (DFG), where
each node is a *model function call* (MFC, e.g., generate, inference, or
train_step called on an LLM) and each edge specifies the data or
parameter version dependency between nodes.

The dataflow graph of PPO looks like:

.. figure:: images/dfg/ppo.svg
   :alt: ppodfg
   :align: center

   The dataflow graph of PPO.

We show another example of the so called `ReMax
<https://arxiv.org/abs/2310.10505>`_ or REINFORCE algorithm below:

.. figure:: images/dfg/reinforce.svg
   :alt: reinforcedfg
   :align: center

   The dataflow graph of ReMax or REINFORCE.

.. note::

   Under this representation, the dataflow graph of pre-training, SFT or
   reward modeling degenerates to a single train_step node.

More examples can be found in the ``docs/images/dfg`` folder. Only data
dependencies are shown in the above figures. Parameter version
dependencies are enforced such that the first call to a model X at step
i is always followed by the last call to X at step i-1.

These figures are plotted by a standalone script
``examples/visualize_dfg.py``. For a given algorithm such as PPO, we
decompose the entire dataflow into individual MFCs, and each MFC is
represented by a :class:`realhf.MFCDef` object. The input and output
keys of each model function call specify their data dependencies. A list
of :class:`realhf.MFCDef` is passed into the ``build_graph`` function to
construct a dataflow graph, which is a instance of ``networkx.DiGraph``.

ReaL can efficiently parallelize the intra-node and inter-node
computation across multiple GPUs. Intra-node parallelization is achieved
by the well-known 3D parallelism strategy, which is usually adopted in
pre-training. Inter-node parallelization represents overlapping or
concurrently executing different nodes on different *device meshes*.

A device mesh D is the unit for executing an individual function call.
It is defined as a two-dimensional grid of GPUs located in the cluster.
The shape of D is denoted as (N, M) if it covers N nodes equipped with M
devices. Note that device meshes with the same shape could have
different locations.

The i-th MFC node will be executed on the device mesh :math:`D_i`. **It
can be either disjoint or overlapping with the device meshes of other
nodes.**

.. note::

   We assume that :math:`D_i` either covers several entire hosts or a
   consecutive portion that is capable of dividing the number of devices
   on one host, e.g., (1, 1), (1, 2), (1, 4), (1, 8), (2, 8), ..., (N,
   8) in a cluster of (N, 8).

The graph building is done in ``realhf/api/core/system_api.py``, during
the post-init of experiment configurations. The concrete definition of
MFCs in different experiments can be found in files under
``realhf/experiments/common/``. All experiment configurations define a
``rpcs`` property, which will be first processed by the
``initial_setup`` method in ``realhf/experiments/common/common.py``,
then passed to the ``ExperimentConfig`` object to build the graph.

************************
 Runtime Infrastrcuture
************************

ReaL implements a worker-based runtime, composed of a single
MasterWorker (MasW) on CPU, and multiple ModelWorkers (ModW), each
occupying a separate GPU. For example, in a cluster of (N, 8), there
will be a single MasW and N * 8 ModWs.

**********
 Overview
**********

Recall that MFCs can have independent (either disjoint or overlapping)
device meshes. From the perspective of a ModW or a GPU, it can host one
or more MFCs. The MasW will run the DFG and send requests to
corresponding handlers. Each request contains the handler name (e.g.,
Actor or Critic), the interface type (e.g., generate or train_step), and
some metadata (e.g., the input and output keys). Upon receiving the
request, the ModW will find the corresponding model, run the
computation, and return results to the MasW to update the dependency.

Inherited from the base Worker class, MasW and ModW run the ``_poll``
method inside a while-loop. The ``_poll`` method is their main job to
do. Outside of the ``_poll`` method, they will listen to the controller
and update their internal states, such that they can be paused, resumed,
or stopped by the controller.

******************************************
 The Procedure of Launching an Experiment
******************************************

This section introduces how ReaL launches experiments through local
subprocess, Ray, or SLURM. Conceptually, the launcher provides a similar
functionality to ``torchrun``, but we did't use ``torchrun`` because
ReaL's code is inherited from the previous SRL project. The scheduler in
SRL can run heterogeneous CPU and GPU tasks, while it's hard to do so
with ``torchrun``.

.. figure:: images/experiment_workflow.svg
   :alt: exp_workflow

   The execution workflow when launching an experiment with ReaL.

ReaL has two levels of configurations. The outer level is based on the
Hydra structured configuration, as we illustrated in the
:doc:`quickstart` section. This level abstracts an experiment into
several configurable fields, which makes the user to conveniently change
the hyperparameters of the experiment, such as the parallelism strategy,
the learning rate, and the batch size.

Then, ReaL will translate the Hydra configuration into a worker-based
configuration. It includes the configurations dataset, model, interface,
and backends to run on each ModW. Please check
``realhf/api/core/config.py`` for concrete examples. The core code of
translation is written in the ``_get_model_worker_configs`` method in
``realhf/experiments/common/common.py``. This configuration level
retains the maximum flexibility. For example, if we need to run some
CPU-heavy tasks as the reward function, we can implement a customized
worker to run the task on CPUs.

The worker configuration is registered as an "experiment" with an unique
name in ``realhf/api/quickstart/entrypoint.py``. Next, it will be
launched by ``realhf.apps.main``. The launcher finds the experiment to
run by its name, load the worker configurations, and submit them to the
scheduler (either SLURM or local subprocesses). The schduler will run a
worker controller to manager the lifetime of other workers. Workers
contiuously check whether there's new message from the controller, and
changes its internal state (e.g., running, pausing, or stopping)
accordingly. After the controller finds that all ModWs and the MasW are
ready, it will send a signal to all workers to start the experiment.
When the schduler finds that some worker is no longer alive, e.g., after
the experiment is done or when an unexpected error occurs, it will
shutdown the controller and all workers, and exit ``realhf.apps.main``.

*******************************************
 Model, Model Interface, and Model Backend
*******************************************

A :class:`Model` is a collection of a transformer-based neural network,
a HuggingFace tokenizer, and some metadata, with an unique name. The
``module`` attribute is usually a ``ReaLModel`` before backend
initialization, and it becomes a :class:`PipelinableEngine` after
backend initialization. ``module`` can be a shard of parameters or even
an empty placeholder when offloading or parameter reallocation is
enabled.

A :class:`ModelInterface` is a collection of concrete implementations
for generation, inference, and training. It doesn't need to implement
them all, e.g., a model for SFT only needs to implement the training
interface. Note that even though the computational workloads can be
categorized into these main types, different algorithms usually have
different side-effects, e.g., PPO requires to compute the GAE during
training while DPO does not. Therefore, we implement interfaces for each
algorithm for easier customization. When the MasW requests a specific
MFC, the ModW will find the correct :class:`Model` and pass it into the
configured algorithm interface for execution. The results returned by
the interface will be sent back to the MasW. This is implemented in the
``__handle_model_function_calls`` method in
``realhf/system/model_worker.py``.

A :class:`ModelBackend` is a functor that wraps the :class:`Model` to
provide additional functionalities like pipelined inference and ZeRO
optimizer. It changes the ``module`` attribute :class:`Model` to a
:class:`PipelinableEngine` object. All interface implementations will
use the APIs of :class:`PipelinableEngine` to run the major computation,
while different interfaces may have different side-effects. The current
backend implementations include ``inference``, which supports pipelined
inference and generate schedule, ``deepspeed``, which supports pipelined
training, ZeRO optimizer, and ZeRO offload, and ``Megatron``, which
supports pipelined training, parameter reallocation, and the distributed
optimizer (similar to ZeRO-1). ZeRO and the distributed optimizer are
implemented in the corresponding external libraries, while ReaL has its
own pipelining, inference-time offload, and parameter reallocation
implementations.

Once launched, the ModW will set up all configured models, interfaces,
and backends. (See the ``__lazy_setup`` method in
``realhf/system/model_worker.py``.) They are indexed by the name of
:class:`Model`. A :class:`MFCDef`, a :class:`Model`, and a
:class:`ModelInterface`, and a :class:`ModelBackend` have one-to-one
relationships in the ModW. Although the interface may integrate many
types of MFC, only the type configured by :class:`MFCDef` will be run
upon this model.

.. note::

   Algorithm customization usually involves implementing a new interface
   for the algorithm. For example, a customized reward interface shown
   in ``examples/customized_exp/ppo_sentiment.py``.

*************************
 MasW-ModW Communication
*************************

The request-reply between the MasW and ModWs is done through ZMQ
sockets. We abstract the communicaion pattern in
``realhf/system/request_reply_stream.py``. The communication channel
will be setup in the ``__lazy_setup`` method in both types of workers.
The communication is lightweight since we only transfer metadata between
them, e.g., the keys and IDs of the input and output tensors. We adopt a
TCP-like protocol to ensure that all involving ModWs receive the request
at the same time. The requests are pushed into a queue in the ModW and
handled sequentially. In addition to MFCs, the requests can also include
initialization, data fetching, saving, evaluation, etc. Please check the
``model_poll_step`` and the ``_poll`` method in
``realhf/system/model_worker.py`` for more details.

***************
 Data Transfer
***************

The dataset resides on the ModWs that are responsible for handling the
source MFC in the DFG. For example, in PPO, the dataset is stored in the
ModWs that handle actor generation. The dataset is sharded across
different data parallel ranks. Check the ``__lazy_setup`` function in
ModW for details.

At the start of each epoch, the MasW will continuously send data
fetching requests to the ModWs until the dataset has been iterated once.
The ModWs will step through the dataloader and return the metadata
(e.g., sequence length, keys in the dataset, IDs, etc) to the MasW. The
MasW will fill these metadata into an internal buffer. This buffer
records how many times this piece of data has been used in the DFG, and
which keys have been produced by MFCs. Once the dependency of a MFC is
satisfied, i.e., the required input keys are all ready in the buffer,
the MasW will send a request to the corresponding ModWs to run the MFC.
If the MFC produces some new keys, the produced GPU tensors will be
stored locally, and ModWs will send the metadata back to the MasW for
updating the buffer. After a piece of data has been used by all nodes in
the DFG, it will be poped out. If the buffer size is too small for
subsequent execution, the MasW will send data fetching requests to the
ModWs for the next epoch. These behaviors are implemented in the
``load_data_func`` in MasW, the ``prefetch_from_dataset`` and
``model_poll_step`` methods in ModW, and ``realhf/system/buffer.py``.

Data is replicated over tensor and pipeline parallel dimensions, and
sharded across the data parallel dimension. Since different MFCs may
have different device meshes and parallel strategies, we need to
transfer the data from the owner (or producer) to the consumer before
MFC computation. This is implemented as **hooks** in requests. Since the
MasW keeps the global information, it can append the source and the
destination of required data in the pre-hooks and send them to related
ModWs. The ModW will trigger GPU-GPU data transfer via a broadcast-based
algorithm to properly get all the required data. This is implemented in
the ``__handle_one_rpc_hook`` method in ModW.

************************
 Parameter Reallocation
************************

ReaL automatically reallocates model parameters to peer GPUs or the CPU
memory to reduce GPU memory usage and the communication volume caused by
parallelization. We must mention a implementation-wise fact first: if a
model is about to be trained, the memory of its parameters cannot be
released after reallocation. This is because the PyTorch optimizer,
e.g., Adam, keeps the model parameter as dictionary keys, and there are
always GPU tensor handles alive.

Due to this limitation, we have to categorize models as trainable or
non-trainable. If any MFC involves training the model, the role of the
model will be categorized as trainable. For example, actor and critic
are trainable in PPO, while the reward and reference models are not.

For non-trainable models, we can safely reallocate their parameters to
the CPU memory, aka offloading. The parameters will be asynchronously
transferred back (i.e., overlapped computation and communication) to the
GPU memory during the next forward pass. When there are multiple
inference requets called upon the same role, they will have their own
copy of parameters and be offloaded independently. Offload is
implemented as the ``async_offload`` method in ``ReaLModel``, called in
the ``__handle_one_rpc_hook`` method in ModW.

For trainable models, if there is also an inference or generate MFC
called upon this role (e.g., Actor and Critic in PPO), we can adopt
different parallel strategies for different MFCs and dynamically
reallocate parameters to reduce communication overhead. The training MFC
holsd its own parameters in GPU memory, while non-training MFCs only
hold empty placeholders. When the non-training MFC is requested, MasW
will append a pre-hook to the request containing all the information for
reallocating the parameters, and a post-hook to revert this operation.
The reallocation is implemented in the ``__handle_one_rpc_hook`` method
in ModW. Note that since the trainable parameters cannot be released,
the reverse reallocation is essentially droping the parameters used for
inference or generating.

.. note::

   The above limitation of PyTorch is not a intrisinc problem. We can
   re-implement the optimizer and use parameter names as keys. However,
   this requires modifying Megatron and DeepSpeed correspondingly and is
   not a trivial task.
