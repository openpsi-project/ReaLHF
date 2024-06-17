Introduction
----------------

ReaL introduces a novel technique called *Parameter Reallocation*
(the name *ReaL* is the abbreviation for *ReaLlocation*), which dynamically
shifts model parameters and changes the parallelization strategy during training.
This technique can significantly reduce the communication overhead and improve
GPU utilization in RLHF training, leading to a substantial speedup over the state-of-the-art
open-source systems.

We observe two major limitations based on our profiling
of the previous RLHF systems, as shown in the :ref:`timeline`.

.. _timeline:

.. figure:: timeline.svg
   :alt: timeline

   Timeline Figure
   
   Execution timelines of ReaL and existing systems based on profiling.

First, when models are distributed
to every GPU node that applies the same parallelization
strategy, such as in DeepSpeed-Chat, it is often over-parallelized.
Over-parallelization leads to
substantial synchronization and communication overhead
(the light purple bars).

An alternative way is to assign different
models to different GPU nodes, where models can execute
concurrently, such as OpenRLHF.
However, our second observation is that such
asymmetric parallelization often causes under-utilization of
the GPUs (e.g., the gray areas) because
of the dependencies between tasks.

The key idea of ReaL is to enable dynamic **reallocation of
model parameters** between GPUs to improve the efficiency of
the entire RLHF training process.
By first choosing a parallelization strategy tailored for
each model function call
(e.g., use pipelining for Generation, while tensor parallelism for Training)
and then executing these calls concurrently with a smaller
parallelization degree (e.g., Actor and Critic in Training),
we can eliminate redundant communication while maximizing GPU utilization,
effectively addressing the limitations of
prior solutions.


We show throughput comparison with the state-of-the-art open-source systems
in the following figure.

.. image:: vws.svg

.. "Scale Actor" maintains the sizes
.. of Critic and Reward at 7B while increasing the sizes of Actor and Reference with the number of GPUs.
.. "Scale Critic" follows the opposite approach, and
.. "Scale Both" increases sizes of all models proportionately.

We also show the estimated time for
completing the entire full-scale 4*70B RLHF training process,
composed of 4 iterations with 400 steps for each iteration as for LLaMA-2.

.. _est_time_table:

+--------------+---------------+---------------+---------------+
|   System     | DeepSpeedChat |   OpenRLHF    |   ReaL-LLM    |
+==============+===============+===============+===============+
| Time (hours) |     141.5     |    152.8      |  **17.0**     |
+--------------+---------------+---------------+---------------+