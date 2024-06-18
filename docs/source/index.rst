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

ReaL proposes a novel *parameter realloaction* technique.
It dynamically shifts parameters and changes parallel strategies
of LLMs during training.
This technique can largely reduce the communication overhead and improve
GPU utilization for RLHF.

Together with the most advanced
techniques for LLM training, such as 3D parallelism,
ZeRO optimization, and offloading,
ReaL can scale RLHF training to
hundreds or thousands of GPUs,
maintaining high throughput and efficiency.

Beyond large-scale training, ReaL is also memory-efficient with limited resources.
For example, ReaL can train 70B LLMs with offloading on a single node.

For more details, check our `introduction page <intro>`_.

**Easy to use**
~~~~~~~~~~~~~~~~~~~~~~~

Install with PyPI or use our Docker image, then
run your experiment with a single command!
Check our `quickstart guide <quickstart>`_ for more details.

**Flexible**
~~~~~~~~~~~~~~~~~~~~~~~

ReaL's system implementations are fully decoupled with
algorithm interfaces. Get the best performance for your
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




