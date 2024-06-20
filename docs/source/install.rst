Installation
==============

Docker Images
--------------

The easiest way to run ReaL is by using the provided Docker images.
We offer a CPU-only image for launching experiments and a runtime GPU
image for deployment in a cluster. The Dockerfile is also available in the repository.

To pull the images, run:

.. code-block:: console

   $ docker pull docker.io/garrett4wade/real-cpu:22.04-0.1.0
   $ docker pull docker.io/garrett4wade/real-gpu:23.10-py3-0.1.0

The CPU image is built from "ubuntu:22.04" and the GPU image is built from "nvcr.io/nvidia/pytorch:23.10-py3". The current package version is "0.1.0".

After pulling the Docker images, run your Docker container locally on a GPU node with the following command:

.. code-block:: console

   $ docker run -it --gpus all garrett4wade/real-gpu:23.10-py3-0.1.0 bash

The source code is available at /realhf inside the container. This is an editable installation, so you can modify the code or run experiments directly.

If you want to develop the code outside a Docker container,
remember to rerun the editable installation command after mounting:

.. code-block:: console

   $ pip install -e /your/mounted/code/path --no-build-isolation


Install From PyPI or Source
----------------------------

If you prefer not to use Docker, you can also install ReaL from PyPI or from the source.

.. note::

   We don't upload a pre-built wheel to PyPI, so the installation will require compiling the C++ and CUDA extensions. If CUDA is not available on your machine, only the C++ extension will be installed.

First, clone the repository and install all dependencies:

.. code-block:: console

   $ pip install -U pip
   $ git clone https://github.com/openpsi-project/ReaLHF
   $ cd ReaLHF
   $ python3 -m pip install -r requirements.txt

On a GPU machine, also install the required runtime packages:

.. code-block:: console

   $ MAX_JOBS=8 python3 -m pip install git+https://github.com/NVIDIA/TransformerEngine.git@v1.4 --no-deps --no-build-isolation
   $ MAX_JOBS=8 python3 -m pip install flash_attn==2.4.2 --no-build-isolation

Install ReaLHF from PyPI:

.. code-block:: console

   $ python3 -m pip install realhf --no-build-isolation

The PyPI package allows you to launch existing experiments with the quickstart command. If you want to modify the code, you should clone the source code and install it from the source:

.. code-block:: console

   $ git clone https://github.com/openpsi-project/ReaLHF
   $ cd ReaLHF
   $ python3 -m pip install -e . --no-build-isolation

Next, check :doc:`quickstart`` for instructions on running experiments.
