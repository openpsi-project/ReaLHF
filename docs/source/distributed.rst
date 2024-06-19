Set Up Distributed Experiments
==================================

Currently, ReaL supports launching distrbited experiments using 
`SLURM <https://slurm.schedmd.com/documentation.html>`_
with the `Pyxis <https://github.com/NVIDIA/pyxis>`_ plugin.
This plugin allows for launching enroot containers with the
``srun`` command.

To set up distributed experiments, you should write a JSON
cluster configuration as the example in ``examples/cluster_config.json``.

- ``cluster_type``: The type of cluster. Currently, only "slurm" is supported.
- ``cluster_name``: The name of the cluster. Arbitrary.
- ``fileroot``: An NFS path that all nodes can access. This is where the log and checkpoints will be stored.
- ``default_mount``: Comma separated list of paths to mount on all nodes. This should include the above ``fileroot``.
- ``node_type_from_node_name``: A dictionary mapping a regular expression to a node type. Any host in this cluster should match one of these regular expressions. Node types include ["g1", "g2", "g8", "a100"]. "g" refers low-end GPUs in the cluster.
- ``gpu_type_from_node_name``: A dictionary mapping a regular expression to a GPU type. GPU type is used by SLURM.
- ``cpu_image``: The docker image of the controller and the master worker.
- ``gpu_image``: The docker image of the model worker.
- ``node_name_prefix``: The prefix of the host names. We assume host names in the cluster is prefixed by a string followed by some integer, e.g., "com-01", where "com-" is the prefix.

The path of this file should be specified in the ``CLUSTER_SPEC_PATH`` environment variable
inside the used docker images and when launching the experiment. For example,

.. code-block:: console

    CLUSTER_SPEC_PATH=/tmp/my-cluster.json python3 -m realrlhf.apps.quickstart ppo ...

You also need to add an additional layer in the docker images like the following:

.. code-block:: dockerfile

    FROM docker.io/garrett4wade/real-cpu
    ENV CLUSTER_SPEC_PATH=/tmp/my-cluster.json