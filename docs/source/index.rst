.. ReaL-LLM documentation master file, created by
   sphinx-quickstart on Mon Jun 10 10:57:12 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ReaL-LLM's documentation!
====================================

   **ReaL is a highly efficient system
   for LLM RLHF training at all scales.**


Highlights
-----------

**Efficient at all scales**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Thanks to open-source frameworks
such as Megatron-LM and DeepSpeed,
ReaL supports the most advanced
techniques for LLM training, such as 3D parallelism,
ZeRO optimization, and offloading.

Together with the proposed *parameter reallocation*
technique, ReaL can scale RLHF training to
hundreds or thousands of GPUs,
maintaining high throughput and efficiency.
On the other extreme, ReaL is also memory-efficient
and capable of training 70B LLMs with offloading on a single node.

**Easy to use**
~~~~~~~~~~~~~~~~~~~~~~~

Install with PyPI or use our Docker image, then
run your experiment with a single command!

**Flexible**
~~~~~~~~~~~~~~~~~~~~~~~

System implementations are fully decoupled with
algorithm interfaces. Get the best performance of your
customized application within 100 lines of code!

Contents
----------------

.. toctree::
   :maxdepth: 3

   intro
   install
   quickstart
   expconfig
   customization
   algo
   arch
   contributing




