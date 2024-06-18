Installation
==============

Docker Images
--------------

The easiest way to run ReaL is to use the provided Docker images.
We provide a CPU-only image to launch experiments and a runtime GPU
image to be deployed in the cluster.
The Dockerfile has been provided in the repository as well.

To pull the images, run:

.. code-block:: console

   $ docker pull garrett4wade/realrlhf-cpu
   $ docker pull garrett4wade/realrlhf-gpu

To build the images from scratch, run:

.. code-block:: console

   $ docker build --target=cpu -t realrlhf-cpu .
   $ docker build --target=gpu -t realrlhf-gpu .

Install From PyPI or Source
----------------------------

If you don't want to use docker, you can also install ReaL from PyPI
or from source.

Install from PyPI:

.. code-block:: console

   $ pip install realrlhf --no-build-isolation

Install from source:

.. code-block:: console

   $ cd /path/to/realrlhf/directory
   $ pip install -e . --no-build-isolation

.. note::

   In an environment without CUDA, ReaL will only
   install necessary Python modules for launching distributed experiments.
   That's why we have two different docker images for
   launching and deploying ReaL.
