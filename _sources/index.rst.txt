.. ReaL documentation master file, created by
   sphinx-quickstart on Mon Jun 10 10:57:12 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ReaL's documentation!
====================================

Highlights of ReaL
-----------

**Super-Efficient**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ReaL introduces a novel *parameter reallocation* technique. It dynamically shifts parameters and 
adjusts parallel strategies of LLMs during training. This technique significantly reduces communication 
overhead and improves GPU utilization for RLHF.

Combined with advanced techniques for LLM training, such as 3D parallelism, ZeRO optimization, and offloading, 
ReaL can scale RLHF training to hundreds or thousands of GPUs, maintaining high throughput and efficiency.

Beyond large-scale training, ReaL is also memory-efficient with limited resources. For example, ReaL can 
train 70B LLMs with offloading on a single node.

For more details, check our `introduction page <intro>`_.

**Easy to use**
~~~~~~~~~~~~~~~~~~~~~~~

Install with PyPI or use our Docker image, then run your experiment with a single command!

Check our `quickstart guide <quickstart>`_ for more details.

**Flexible**
~~~~~~~~~~~~~~~~~~~~~~~

ReaL's system implementations are fully decoupled from algorithm interfaces. Achieve optimal performance 
for your customized application within 100 lines of code!

Please refer to our `customization guide <customization>`_ for more details.

Contents
----------------

.. toctree::
   :maxdepth: 3

   intro
   install
   quickstart
   expconfig
   customization
   arch
   distributed
   contributing




