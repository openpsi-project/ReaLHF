Installation
==============

Docker Images
--------------

The easiest way to run ReaL is by using the provided Docker images.
We offer a CPU-only image for launching experiments and a runtime GPU
image for deployment in a cluster. The Dockerfile is also available in the repository.

To pull the images, run:

.. code-block:: console

   $ export REAL_VERSION="0.1.0"
   $ docker pull docker.io/garrett4wade/real-cpu:22.04-${REAL_VERSION}
   $ docker pull docker.io/garrett4wade/real-gpu:23.10-py3-${REAL_VERSION}

The CPU image is built from "ubuntu:22.04" and the GPU image is built from "nvcr.io/nvidia/pytorch:23.10-py3".
The current package version is "0.1.0".

After pulling the Docker images, run your Docker container locally on a GPU node with the following command:

.. code-block:: console

   $ docker run -it --gpus all garrett4wade/real-gpu:23.10-py3-0.1.0 bash

The source code is available at /realhf inside the container.
This is an editable installation, so you can modify the code or run experiments directly.

If you want to develop the code outside a Docker container,
you should mount the code directory to the container, e.g.,

.. code-block:: console

   $ docker run -it --gpus all --mount type=bind,src=/path/outside/container,dst=/realhf garrett4wade/real-gpu:23.10-py3-0.1.0 bash

If your destination path is not /realhf,
remember to rerun the editable installation command after mounting:

.. code-block:: console

   $ REAL_CUDA=1 pip install -e /your/mounted/code/path --no-build-isolation

.. note::

   The ``REAL_CUDA`` environment variable is used to install the CUDA extension.


Install From PyPI or Source
----------------------------

If you prefer not to use the provided Docker image,
you can also start with an image provided by NVIDA (e.g., nvcr.io/nvidia/pytorch:23.10-py3)
and install ReaL from PyPI or from the source.

.. note::

   We don't upload a pre-built wheel to PyPI, so the installation will require compiling the C++ and CUDA extensions.
   Control whether to install the extentions with environment variables ``REAL_CUDA`` and ``REAL_NO_EXT``.

   The CUDA extention will be installed only if ``REAL_CUDA`` is set to 1.
   No extention will be installed if ``REAL_NO_EXT`` is set to 1.

First, clone the repository and install all dependencies:

.. code-block:: console

   $ pip install -U pip
   $ git clone https://github.com/openpsi-project/ReaLHF
   $ cd ReaLHF
   $ pip install -r requirements.txt

On a GPU machine, also install the required runtime packages:

.. code-block:: console

   $ export MAX_JOBS=8  # Set the number of parallel jobs for compilation.
   $ pip install git+https://github.com/NVIDIA/TransformerEngine.git@v1.4 --no-deps --no-build-isolation
   $ pip install flash_attn==2.4.2 --no-build-isolation

.. note::

   ``MAX_JOBS`` sets the number of parallel jobs for compilation.
   A larger value will consume more memory (and potentially lead to OOM stuck) and CPU resources.
   Adjust the value according to your machine's specifications.

Install ReaLHF from source (recommended, for the latest build):

.. code-block:: console

   $ git clone https://github.com/openpsi-project/ReaLHF
   $ cd ReaLHF
   $ REAL_CUDA=1 pip install -e . --no-build-isolation

Or install from PyPI (for stable build):

.. code-block:: console

   $ REAL_CUDA=1 pip install realhf --no-build-isolation

The PyPI package allows you to launch existing experiments with the quickstart command.
If you want to modify the code, you must clone the source code and install it from the source.

Next, check :doc:`quickstart` for instructions on running experiments.
