##############
 Contributing
##############

..
   This repository is developed and maintained by `Wei Fu <garrett4wade.github.io>`_

..
   and `Zhiyu Mei <https://openreview.net/profile?id=~Zhiyu_Mei1>`_, both of whom are

..
   PhD students at `IIIS, Tsinghua University <https://iiis.tsinghua.edu.cn/en/>`_

..
   advised by Professor `Yi Wu <https://jxwuyi.weebly.com/>`_.

..
   We acknowledge that due to limited time and resources,

..
   the quality of the documentation and code in this repository is not very high.

..
   As a result, it can be quite challenging for potential developers to

..
   read the code and contribute new features.

If you wish to contribute to this repository or have any questions about
the code, please do not hesitate to raise issues or contact us directly.
We will do our best to assist you. Currently, there is no template for
issues or pull requests.

We hope the open-source community can help improve this repository and
enable RLHF technology to truly empower the applications of LLM.

***************
 Documentation
***************

The source code is documented using Sphinx in the ``docs`` folder. On a
node with docker-compose installed, run

.. code:: bash

   make docs

Then the documentation will be available at ``http://localhost:7780``.

Every time the documentation files are changed, you should run the above
command to update the documentation.

The GitHub Pages will be updated automatically after the PR is merged.

************
 Formatting
************

.. code:: bash

   # For .py files
   docformatter -i ${FILE} && isort ${FILE} && black -q ${FILE}
   # For C/C++ files
   clang-format -i ${FILE}
   # For documentation
   rstfmt docs

*********
 Testing
*********

.. code:: bash

   # Run CPU tests
   pytest -m "not gpu"
   # Run CPU tests and GPU tests that require a single GPU
   pytest -m "not distributed"
   # On a node with multiple GPUs, run all tests
   pytest

************************
 Building Docker Images
************************

.. code:: bash

   # Build the GPU image
   docker build -t real-gpu:24.03-0.3.0 -f Dockerfile --target gpu --build-arg REAL_GPU_BASE_IMAGE=nvcr.io/nvidia/pytorch:24.03-py3 --build-arg REAL_CPU_BASE_IMAGE=ubuntu:22.04 .
   # Build the CPU image
   docker build -t real-cpu:22.04-0.3.0 -f Dockerfile --target cpu --build-arg REAL_GPU_BASE_IMAGE=nvcr.io/nvidia/pytorch:24.03-py3 --build-arg REAL_CPU_BASE_IMAGE=ubuntu:22.04 .
