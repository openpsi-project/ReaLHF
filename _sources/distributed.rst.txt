Set Up Distributed Experiments
==================================

Currently, ReaL supports launching distributed experiments using
`SLURM <https://slurm.schedmd.com/documentation.html>`_
with the `Pyxis <https://github.com/NVIDIA/pyxis>`_ plugin.
This plugin allows for launching enroot containers with the
``srun`` command.

To set up distributed experiments, you need to create a JSON
cluster configuration file, as shown in the example in  ``examples/cluster_config.json``.

- ``cluster_type``: The type of the cluster. Currently, only "slurm" is supported.
- ``cluster_name``: The name of the cluster. Arbitrary.
- ``fileroot``: An NFS path accessible by all nodes. This is where logs and checkpoints will be stored.
- ``default_mount``: A comma-separated list of paths to mount on all nodes. This should include the ``fileroot`` mentioned above..
- ``node_type_from_node_name``: A dictionary mapping a regular expression to a node type. Every host in this cluster should match one of these regular expressions. Node types include ["g1", "g2", "g8", "a100"]. "g" refers to low-end GPUs in the cluster.
- ``gpu_type_from_node_name``: A dictionary mapping a regular expression to a GPU type. The GPU type is used by SLURM.
- ``cpu_image``: The Docker image for the controller and the master worker.
- ``gpu_image``: The Docker image for the model worker.
- ``node_name_prefix``: The prefix of the host names. We assume that host names in the cluster are prefixed by a string followed by an integer, e.g., "com-01", where "com-" is the prefix.

The path of this file should be specified in the ``CLUSTER_SPEC_PATH`` environment variable
inside the Docker images used and when launching the experiment. For example:

.. code-block:: console

    CLUSTER_SPEC_PATH=/tmp/my-cluster.json python3 -m realhf.apps.quickstart ppo ...

You also need to add an additional layer in the Docker images as shown below:

.. code-block:: dockerfile

    FROM garrett4wade/real-cpu:22.04-0.1.0
    ENV CLUSTER_SPEC_PATH=/tmp/my-cluster.json