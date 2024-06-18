Quickstart
===========

Installation
---------------

First, clone the ReaL repository from GitHub\:

.. code-block:: shell

    $ git clone xxx
    $ cd xxx

RLHF with 4x LLaMA-7B in One Hour
------------------------------------------------

If you are not familar with the procedure of RLHF,
please refer to the `InstrctGPT paper <https://arxiv.org/abs/2203.02155>`_.
This tutorial will go over the main stages of RLHF,
including SFT, reward modeling, and DPO/PPO.

We also provide sample datasets for each stage.

Stage 1: Supervised Fine-Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prepare your customized dataset in a json or jsonl format,
where each entry should be a dictionary with two keys, "prompt" and "answer".
For example,

.. code-block:: json

    {"prompt": "What is the capital of France?", "answer": "The capital of France is ..."}
    {"prompt": "Please make a travel plan for visiting Berlin?", "answer": "..."}

.. note::

    If you haved not prepared a dataset for your application, you can download the our
    `sample dataset <https://drive.google.com/drive/folders/1xWIJ9DRLNQZxDrkCfAPE12euLLuWQGE-?usp=sharing>`_
    to go through this tutorial.
    The sample dataset is used for controlled sentiment generation,
    where the LLM should learn to generate positive movie comments given a context.

    ``sft_pos-train.jsonl`` and ``sft_pos-valid.jsonl`` are the training and validation sets for SFT, respectively.

Then, run the following command to fine-tune the model on your dataset:

.. code-block:: shell

    $ python3 -m realrlhf.apps.quickstart sft \
        experiment_name=quickstart-sft \
        trial_name=my-trial \
        allocation_mode=manual \
        mode=local \
        total_train_epochs=8 \
        save_freq_steps=50 eval_freq_epochs=1 \
        model.type._class=llama \
        model.type.size=7 \
        model.type.is_critic=False \
        model.path=/path/or/HF-identifier/of/llama \
        model.gradient_checkpointing=True \
        model.optimizer.type=adam \
        dataset.train_path=/path/to/train/dataset.jsonl \
        dataset.valid_path=/path/to/valid/dataset.jsonl \
        dataset.max_seqlen=1024 \
        allocation.parallel.pipeline_parallel_size=1 \
        allocation.parallel.model_parallel_size=2 \
        allocation.parallel.data_parallel_size=4 \
        allocation.parallel.use_sequence_parallel=True \
        dataset.train_bs_n_seqs=512 \
        dataset.valid_bs_n_seqs=512

ReaL adopts `structured configurations <https://hydra.cc/docs/tutorials/structured_config/intro/>`_
in `Hydra <https://hydra.cc/>`_ to manage command line options.
The options in the above command correspond to a Python
dataclass object :class:`realrlhf.SFTConfig`.
The attributes, including the model type, the learning rate, and the parallel strategy,
can be recursively overwritten via command line options.
Please check :doc:`expconfig` for more details.

.. note::
    As a kind reminder, the passed-in value should be `null` to represent `None` in python.

Importantly, the user should choose an appropriate parallel strategy
as well as a moderate batch size according to the hardware setting.
In the given example, the experiment will use 8 GPUs in total,
with parallel strategy (pipe=1, tensor=2, data=4) and a batch size of 512.

After the experiment has been successfully launched,
you will see the training logs in the console like this\:

.. code-block:: console

    xxxx

The above output prompts the log and the checkpoint paths of this experiment,
according to the given ``experiment_name`` and ``trial_name``.

.. note::

    ReaL loads directly from HuggingFace models and also saves checkpoints
    as HuggingFace models, which makes it convinent to use the pre-trained models
    and to deploy trained models with inference engines like vLLM.

The SFT experiment will take about xx minutes to finish using our provided dataset.
Let's move on to the next stage.

Stage 2: Reward Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prepare your customized dataset in a json or jsonl format,
where each entry should be a dictionary with three keys,
"prompt", "pos_answer", and "neg_answers".

"prompt" should be a string, while "pos_answer" and "neg_answers"
should be lists of strings with the same size, forming pairwise comparisons.

.. note::

    If you haved not prepared a dataset for your application, you can download the our
    `sample dataset <https://drive.google.com/drive/folders/1xWIJ9DRLNQZxDrkCfAPE12euLLuWQGE-?usp=sharing>`_
    to go through this tutorial.
    The sample dataset is used for controlled sentiment generation,
    where the LLM should learn to generate positive movie comments given a context.

    ``rm_paired-train.jsonl`` and ``rm_paired-valid.jsonl`` are the 
    training and validation sets for reward modeling, respectively.


.. code-block:: shell

    $ python3 -m realrlhf.apps.quickstart rw \
        experiment_name=quickstart-rw \
        trial_name=my-trial \
        mode=local \
        allocation_mode=manual \
        total_train_epochs=1 \
        save_freq_steps=5 eval_freq_epochs=1 \
        model.type._class=llama \
        model.type.size=7 \
        model.type.is_critic=True \
        model.path=/saved/sft/model/path \
        allocation.parallel.pipeline_parallel_size=2 \
        allocation.parallel.model_parallel_size=2 \
        allocation.parallel.data_parallel_size=2 \
        allocation.parallel.use_sequence_parallel=True \
        model.gradient_checkpointing=True \
        dataset.train_path=/path/to/train/dataset.jsonl \
        dataset.valid_path=/path/to/valid/dataset.jsonl \
        dataset.max_pairs_per_prompt=2 \
        dataset.max_seqlen=1024 \
        dataset.train_bs_n_seqs=512 \
        dataset.valid_bs_n_seqs=512

It's a common practice to use the SFT model to initialize the reward model.
Therefore, we can pass the path of the saved SFT model as the ``model.path`` option.
In reward modeling, the batch size is the number of paired comparisons.
With a batch size of 512, there will be 512 positive samples and 512 negative samples in each batch.


Training the reward model until convergence can be very fast.
In the given example, we can preemptively stop the training after 15 steps, which approximately takes xxx minutes.

Stage 3.1: Direct Preference Optimization (DPO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Besides the ordinary RLHF procedure with PPO,
ReaL also supports the DPO algorithm, which avoids reward modeling.

The dataset for DPO is exactly the same as reward modeling.

.. code-block:: shell

    $ python3 -m realrlhf.apps.quickstart dpo \
        experiment_name=quickstart-dpo \
        trial_name=my-trial \
        allocation_mode=manual \
        mode=local \
        total_train_epochs=2 \
        save_freq_steps=5 \
        actor.type._class=llama \
        actor.type.size=7 \
        actor.type.is_critic=False \
        actor.path=/saved/sft/model/path \
        actor_train.parallel.pipeline_parallel_size=1 \
        actor_train.parallel.model_parallel_size=4 \
        actor_train.parallel.data_parallel_size=2 \
        actor_train.parallel.use_sequence_parallel=True \
        ref.type._class=llama \
        ref.type.size=7 \
        ref.type.is_critic=False \
        ref.path=/saved/sft/model/path \
        ref_inf.parallel.pipeline_parallel_size=1 \
        ref_inf.parallel.model_parallel_size=2 \
        ref_inf.parallel.data_parallel_size=4 \
        ref_inf.parallel.use_sequence_parallel=True \
        dataset.train_path=/path/to/train/dataset.jsonl \
        dataset.max_pairs_per_prompt=2 \
        dataset.max_seqlen=1024 \
        dataset.train_bs_n_seqs=512 \
        dataset.valid_bs_n_seqs=512

Note that there's a major difference between DPO and SFT or reward modeling.
DPO involves two different models, the *actor* and the *reference*.
The former is the primary LLM to be trained and the latter is the freezed SFT
model to provide KL regularizations.

A training iteration of DPO is composed of two steps\:

- *RefInf*\: The reference model performs a forward step to compute the log probabilities of positive and negative answers.

- *ActorTrain*\: Given the reference log probabilities, the actor model computes the DPO loss, run the backward pass, and update the parameters.

In ReaL, these two steps can run with different parallel strategies, which allows
maximizing efficiency of the individual workloads.
For example, pipelined inference can be faster than tensor-paralleled inference due to
the reduced communication overhead.
These parallel strategies can be specified in the ``ref_inf`` and the ``actor_train`` fields.

What's more, ReaL can automatically *offload* the parameters of the reference model once *RefInf* is done.
This does not require any additional configurations.
Consequently, **ReaL's DPO is as memory-efficient as training a single model like SFT!**


Stage 3.2: PPO
~~~~~~~~~~~~~~~~~

After the SFT and reward modeling stages, we can proceed to the PPO stage.
The dataset for PPO should be a json or jsonl file with each entry being a dictionary of a single key "prompt".

.. note::

    If you haved not prepared a dataset for your application, you can download the our
    `sample dataset <https://drive.google.com/drive/folders/1xWIJ9DRLNQZxDrkCfAPE12euLLuWQGE-?usp=sharing>`_
    to go through this tutorial.
    The sample dataset is used for controlled sentiment generation,
    where the LLM should learn to generate positive movie comments given a context.

    ``ppo_prompt.jsonl`` is the training set for PPO.

.. code-block:: shell

    $ python3 -m realrlhf.apps.quickstart ppo \
        experiment_name=quickstart-ppo \
        trial_name=my-trial \
        total_train_epochs=4 \
        allocation_mode=heuristic \
        save_freq_steps=null \
        actor.type._class=llama \
        actor.type.size=7 \
        actor.type.is_critic=False \
        actor.path=/saved/sft/model/path \
        actor.gradient_checkpointing=True \
        critic.type._class=llama \
        critic.type.size=7 \
        critic.type.is_critic=True \
        critic.path=/saved/rw/model/path \
        critic.gradient_checkpointing=True \
        ref.type._class=llama \
        ref.type.size=7 \
        ref.type.is_critic=False \
        ref.path=/saved/sft/model/path \
        rew.type._class=llama \
        rew.type.size=7 \
        rew.type.is_critic=True \
        rew.path=/saved/rw/model/path \
        dataset.path=/path/to/prompt/dataset.jsonl \
        dataset.max_prompt_len=256 \
        dataset.train_bs_n_seqs=128 \
        ppo.max_new_tokens=256 \
        ppo.min_new_tokens=256 \
        ppo.ppo_n_minibatches=4 \
        ppo.kl_ctl=0.1 \
        ppo.force_no_logits_mask=False \
        ppo.value_eps_clip=0.2 \
        ppo.reward_output_scaling=10.0 \
        ppo.adv_norm=True ppo.value_norm=True \
        ppo.top_p=0.9 ppo.top_k=1000

The configuration options of PPO is the most complicated one among the three stages.
PPO involves four different models, namely *Actor*, *Critic*, *Reference*, and *Reward*.
Each individual model can have different functionalities across a training iteration.
For example, the *Actor* should first *generate* responses given prompts and then
be *trained* given rewards, values, and KL regularizations.

Training iterations of PPO can be illustrated as follows:

.. image:: images/rlhf_dfg.svg
    :alt: Dataflow graph of RLHF.
    :align: center

We can see that there are six distinct *function calls* on these four models.
In ReaL, these function calls can have independent allocations and parallel strategies.
Between two function calls upon the same model, ReaL will automatically re-allocate
model parameters between source and destination locations and properly remap
parallel strategies.
This feature can substantically reduce communication overhead caused by parallelization
and improve GPU utilization.
Please check :doc:`intro` for more details.

In the above command, fields ``actor``, ``critic``, ``ref``, and ``rew`` specify the configurations of the four models.
The allocations and parallel strategies for function calls are automatically
handled by the ``heuristic`` allocation mode.
This is a near-optimal execution strategy found by the search engine in ReaL.

For the details of PPO hyperparameters in the ``ppo`` field, please check
:class:`realrlhf.PPOHyperparameters` for detailed explaination.


We train PPO on 5000 prompts over 4 epochs, which consumes about xxx minutes.
Summing up the time of the three stages, we can finish the RLHF process with ReaL
in just one hour.
This efficiency can largely help algorithm developers to search for the best hyperparameters
and to iterate on the algorithm design.
