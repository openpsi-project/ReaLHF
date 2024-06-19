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

   $ docker pull docker.io/garrett4wade/real-cpu
   $ docker pull docker.io/garrett4wade/real-gpu

.. warning::

   when using these docker images locally, the user should mount the user code directory
   to path ``/realhf`` in the container. This is because the image shifts an editable
   installation at ``/realhf``. When the user code overwrites this path, the change of user
   code will take effect without re-installing this ``realhf`` PyPI package.

   It's also okay to mount to another location and re-install the package in the container.

To build the images from scratch, run:

.. code-block:: console

   $ docker build --target=cpu -t real-cpu .
   $ docker build --target=gpu -t real-gpu .

Install From PyPI or Source
----------------------------

If you don't want to use docker, you can also install ReaL from PyPI
or from source.

Install from PyPI:

.. code-block:: console

   $ pip install realhf --no-build-isolation

.. note::

   Installing from the PyPI wheel still requires the user to clone the
   source code to launch experiments.

Install from source:

.. code-block:: console

   $ $ git clone https://github.com/openpsi-project/ReaLHF
   $ cd ReaLHF
   $ pip install -e . --no-build-isolation

.. note::

   In an environment without CUDA, ReaL will only
   install necessary Python modules for launching distributed experiments.
   That's why we have two different docker images for
   launching and deploying ReaL.
