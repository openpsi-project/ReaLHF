Installation
==============

In an environment without CUDA, ReaL will only
import necessary Python modules for launching distributed experiments.
Customized CUDA kernels will not be installed.

Docker Images
--------------

The easiest way to run ReaL is to use the provided Docker images.
We provide a CPU-only image to launch experiments and a runtime GPU
image to be deployed in the cluster.
The Dockerfile has been provided in the repository as well.

To pull the images, run:

.. code-block:: console

   $ docker pull garrett4wade/reallm-cpu
   $ docker pull garrett4wade/reallm-gpu

To build the images from scratch, run:

.. code-block:: console

   $ docker build --target=cpu -t reallm-cpu .
   $ docker build --target=gpu -t reallm-gpu .

Install From PyPI or Source
----------------------------

If you don't want to use docker, you can also install ReaL from PyPI
or from source.

Install from PyPI:

.. code-block:: console

   $ pip install reallm --no-build-isolation

Install from source:

.. code-block:: console

   $ cd /path/to/reallm/directory
   $ pip install -e . --no-build-isolation

