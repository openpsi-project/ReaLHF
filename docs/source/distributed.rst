####################################
 Setting Up Distributed Experiments
####################################

Currently, ReaL supports launching distributed experiments using `SLURM
<https://slurm.schedmd.com/documentation.html>`_ with the `Pyxis
<https://github.com/NVIDIA/pyxis>`_ plugin, or using `Ray
<https://docs.ray.io/en/latest/index.html>`_. We recommend Ray for its
simplicity and flexibility.

The ``requirements.txt`` file includes the Ray package, and it should be
installed by following our instructions in the :doc:`install` section.
Pyxis and SLURM, however, need to be installed and set up separately by
following their respective documentation.

**************
 Cluster Spec
**************

Before running experiments, ReaL needs to know some information about
the cluster. You need to create a JSON cluster configuration file, as
shown in the example in ``examples/cluster_config.json``.

.. note::

   This file is necessary for SLURM, but optional for Ray.

-  ``cluster_type``: The type of the cluster, either "slurm" or "ray".

-  ``cluster_name``: The name of the cluster. This can be arbitrary.

-  ``fileroot``: An NFS path accessible by all nodes, where logs and
   checkpoints will be stored. By default, ReaL assumes that the /home/
   directory is shared by all nodes in the cluster. ReaL will not run
   correctly if this path is not accessible by all nodes.

-  ``default_mount``: A comma-separated list of paths to mount on all
   nodes, including the fileroot mentioned above. This is only used by
   SLURM.

-  ``node_type_from_node_name``: A dictionary mapping a regular
   expression to a node type. Every host in this cluster should match
   one of these regular expressions. Node types include ["g1", "g2",
   "g8", "a100"], where "g" refers to low-end GPUs in the cluster. This
   is only used by SLURM.

-  ``gpu_type_from_node_name``: A dictionary mapping a regular
   expression to a GPU type. This is used by SLURM.

-  ``cpu_image``: The Docker image for the controller and the master
   worker. This is only used by SLURM.

-  ``gpu_image``: The Docker image for the model worker. This is only
   used by SLURM.

-  ``node_name_prefix``: The prefix of the host names. We assume that
   host names in the cluster are prefixed by a string followed by an
   integer, e.g., "com-01", where "com-" is the prefix. If not provided,
   the default prefix is "NODE", i.e., NODE01, NODE02, etc.

After creating this file, specify it using the ``CLUSTER_SPEC_PATH``
environment variable when launching experiments. For example:

.. code:: console

   CLUSTER_SPEC_PATH=/tmp/my-cluster.json python3 -m realhf.apps.quickstart ppo ...

.. note::

   If the ``CLUSTER_SPEC_PATH`` variable is not set, the Ray mode will
   use a default fileroot ``/home/$USER/.cache/realhf`` and a default
   node prefix "NODE".

   This means that logs and checkpoints will be saved to
   ``/home/$USER/.cache/realhf`` and that the user should specify manual
   allocations like ``NODE[01-02]``.

   It is the user's responsibility to ensure that the fileroot
   ``/home/$USER/.cache/realhf`` is accessible by all nodes in the
   cluster.

**********************************
 Distributed Experiments with Ray
**********************************

Assume you have installed Ray via PyPI on every node in your cluster.
Ensure that the version of Ray is the same on all nodes. Next, we
recommend `setting up the Ray cluster with its CLI
<https://docs.ray.io/en/latest/ray-core/starting-ray.html#start-ray-cli>`_.
You can do this by running the following command:

.. code:: console

   $ # On the head node
   $ ray start --head --port=6379
   Local node IP: xxx.xxx.xxx.xxx

   --------------------
   Ray runtime started.
   --------------------

   Next steps
   To add another node to this Ray cluster, run
       ray start --address='xxx.xxx.xxx.xxx:6379'

   To connect to this Ray cluster:
       import ray
       ray.init()

   To terminate the Ray runtime, run
       ray stop

   To view the status of the cluster, use
       ray status

.. code:: console

   $ # On the worker nodes
   $ ray start --address='xxx.xxx.xxx.xxx:6379'

After setting up the Ray cluster, you can run experiments on the head
node by replacing ``mode=local`` with ``mode=ray`` in the scripts. Now
you can change ``n_nodes``, the device allocation, and parallel
strategies to scale up the experiments with more than
``n_gpus_per_node`` GPUs. No additional changes are required!

We would like to append a few notes on the Ray cluster setup.

Ray Resources
=============

If your cluster is not homogeneous, for example, if the head node is a
CPU machine without a GPU, you can specify the resources using the Ray
CLI:

.. code:: console

   # In the head node
   $ ray start --head --port=6379 --num-cpus=1 --num-gpus=0 --mem=10000

This command will allocate 1 CPU core, 0 GPUs, and 10GB of memory for
the head node. As a result, model workers and the master worker will not
be scheduled on the head node.

ReaL will detect all available resources by calling ``ray.init()`` on
the head node. The driver process that calls ``ray.init()`` does not
consume any resources. Only the workers (i.e., model workers and the
master worker) will consume resources according to the scheduling setup
in the experiment configuration.

If there are not enough resources available, Ray jobs will wait until
the requested resources become available, and ReaL will prompt a message
in the terminal. You can elastically add new nodes with ``ray start`` in
the cluster to increase resources.

Graceful Shutdown
=================

Nodes in the Ray cluster can be shut down with the ``ray stop`` command.
Currently, ReaL has an issue where, when the experiment terminates, it
only kills the driver process, leaving the worker processes stale on
remote nodes.

.. note::

   **Users should manually kill the worker processes on the remote nodes
   using `ray stop`; otherwise, a new experiment on the same Ray cluster
   will get stuck.**

********************************************
 Distributed Experiments with SLURM + Pyxis
********************************************

After specifying the cluster configuration file, you can run experiments
with ``mode=slurm`` in the scripts. ReaL's scheduler will submit jobs to
the SLURM resource manager automatically.
