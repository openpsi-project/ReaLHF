###############
 Core Concepts
###############

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
MasterWorker on CPU, and multiple ModelWorkers, each occupying a
separate GPU. For example, in a cluster of (N, 8), there will be a
single master worker and N * 8 model workers.

**********
 Overview
**********

Recall that MFCs can have independent (either disjoint or overlapping)
device meshes. From the perspective of a model worker or a GPU, it can
host one or more MFCs. The master worker will run the DFG and send
requests to model workers. Each request contains the handler name (e.g.,
Actor or Critic), the interface type (e.g., generate or train_step), and
some metadata (e.g., the input and output keys). Upon receiving the
request, the model worker will find the corresponding model, run the
computation, and return results to the master worker to update the
dependency.

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
and backends to run on each model worker. Please check
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
accordingly. After the controller finds that all model workers and the
master worker are ready, it will send a signal to all workers to start
the experiment. When the schduler finds that some worker is no longer
alive, e.g., after the experiment is done or when an unexpected error
occurs, it will shutdown the controller and all workers, and exit
``realhf.apps.main``.
