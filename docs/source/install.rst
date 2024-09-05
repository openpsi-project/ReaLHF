##############
 Installation
##############

***************
 Docker Images
***************

The easiest way to run ReaL is by using the provided Docker images. We
offer a CPU-only image for launching experiments and a runtime GPU image
for deployment in a cluster. The Dockerfile is also available in the
repository.

To pull the images, run:

.. code:: console

   $ docker pull docker.io/garrett4wade/real-cpu:22.04-0.3.0
   $ docker pull docker.io/garrett4wade/real-gpu:24.03-py3-0.3.0

The CPU image is built from "ubuntu:22.04" and the GPU image is built
from "nvcr.io/nvidia/pytorch:24.03-py3". You can check the latest docker
image version `here
<https://hub.docker.com/r/garrett4wade/real-gpu/tags>`_.

After pulling the Docker images, run your Docker container locally on a
GPU node with the following command:

.. code:: console

   $ docker run -it --rm --gpus all --mount type=bind,src=/path/outside/container,dst=/realhf garrett4wade/real-gpu:24.03-py3-0.3.0 bash

There is an editable installation at ``/realhf`` inside the container,
so your change to the code outside the container should automatically
takes effect.

*****************************
 Install From PyPI or Source
*****************************

If you prefer not to use the provided Docker image, you can also start
with an image provided by NVIDA (e.g.,
``nvcr.io/nvidia/pytorch:24.03-py3``) and install ReaL from PyPI or from
the source.

.. note::

   We don't upload a pre-built wheel to PyPI, so the installation will
   require compiling the C++ and CUDA extensions. Control whether to
   install the extentions with environment variables ``REAL_CUDA`` and
   ``REAL_NO_EXT``.

   The CUDA extention will be installed only if ``REAL_CUDA`` is set to
   1. No extention will be installed if ``REAL_NO_EXT`` is set to 1.

   If you don't want to compile the extensions, please use the provided
   Docker images.

First, clone the repository and install all dependencies:

.. code:: console

   $ pip install -U pip
   $ git clone https://github.com/openpsi-project/ReaLHF
   $ cd ReaLHF
   $ pip install -r requirements.txt

On a GPU machine, also install the required runtime packages:

.. code:: console

   $ export MAX_JOBS=8  # Set the number of parallel jobs for compilation.
   $ pip install git+https://github.com/NVIDIA/TransformerEngine.git@v1.8 --no-deps --no-build-isolation
   $ pip install flash_attn==2.4.2 --no-build-isolation
   $ pip3 install git+https://github.com/tgale96/grouped_gemm.git@v0.1.4 --no-build-isolation --no-deps  # For MoE

.. note::

   ``MAX_JOBS`` sets the number of parallel jobs for compilation. A
   larger value will consume more memory (and potentially lead to OOM
   stuck) and CPU resources. Adjust the value according to your
   machine's specifications.

Install ReaLHF from source (recommended, for the latest build):

.. code:: console

   $ git clone https://github.com/openpsi-project/ReaLHF
   $ cd ReaLHF
   $ REAL_CUDA=1 pip install -e . --no-build-isolation

Or install from PyPI (for stable build):

.. code:: console

   $ REAL_CUDA=1 pip install realhf --no-build-isolation

The PyPI package allows you to launch existing experiments with the
quickstart command. If you want to modify the code, you must clone the
source code and install it from the source.

Next, check :doc:`quickstart` for instructions on running experiments.
