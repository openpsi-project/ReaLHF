Quickstart
===========

Installation
---------------

First, please follow the :doc:`installation instructions <install>`
to install the `reallm` package.

Reproducing RLHF on 7B LLaMA in One Hour
-----------------------------------------

Stage 1: Supervised Fine-Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first stage of RLHF is supervised fine-tuning over an instruction dataset.
Prepare your customized dataset in a json or jsonl format,
where each entry should be a dictionary with two keys, "prompt" and "answer".
For example,

.. code-block:: json

    {"prompt": "What is the capital of France?", "answer": "The capital of France is ..."}
    {"prompt": "Please make a travel plan for visiting Berlin?", "answer": "..."}

Then, run the following command to fine-tune the model on your dataset:

.. code-block:: shell

    $ python3 -m reallm.apps.quickstart sft \
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
in `hydra <https://hydra.cc/>`_ to manage command line options.
The above command launches an SFT experiment, which corresponds to a Python
dataclass object `experiments.common.sft_exp.SFTConfig`.
The members of this object can also be dataclasses.
These attributes can be recursively overwritten via command line options.

Most importantly, we specify the parallel strategy of this SFT model via
the ``allocation`` attribute.
ReaL supports many features from existing libraries,
including 3D parallelism, sequence parallelism, DeepSpeed ZeRO-3,
parameter and optimizer offload, etc.
For all supported options, please check :doc:`expconfig` for details.

ReaL loads directly from HuggingFace models and also saves checkpoints
as HuggingFace models.

Stage 2: Reward Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    $ python3 -m reallm.apps.quickstart rw \
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

Stage 3.1: Direct Preference Optimization (DPO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    $ python3 -m reallm.apps.quickstart dpo \
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

Stage 3.2: PPO
~~~~~~~~~~~~~~~~~

.. code-block:: shell

    $ python3 -m reallm.apps.quickstart ppo \
        experiment_name=debug-quickstart-ppo \
        trial_name=20240613-1 \
        total_train_epochs=4 \
        allocation_mode=manual \
        save_freq_steps=null \
        actor.type._class=llama \
        actor.type.size=7 \
        actor.type.is_critic=False \
        actor.path=$SFT_MODEL_PATH \
        actor.gradient_checkpointing=True \
        critic.type._class=llama \
        critic.type.size=7 \
        critic.type.is_critic=True \
        critic.path=$RW_MODEL_PATH \
        critic.gradient_checkpointing=True \
        ref.type._class=llama \
        ref.type.size=7 \
        ref.type.is_critic=False \
        ref.path=$SFT_MODEL_PATH \
        rew.type._class=llama \
        rew.type.size=7 \
        rew.type.is_critic=True \
        rew.path=$RW_MODEL_PATH \
        dataset.path=/lustre/fw/datasets/imdb/rl/ppo_prompt.jsonl \
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
        ppo.top_p=0.9 ppo.top_k=1000 \
        actor_train.parallel.model_parallel_size=8 \
        actor_train.parallel.pipeline_parallel_size=1 \
        actor_train.parallel.data_parallel_size=1 \
        actor_gen.parallel.model_parallel_size=1 \
        actor_gen.parallel.pipeline_parallel_size=2 \
        actor_gen.parallel.data_parallel_size=4 \
        critic_train.parallel.model_parallel_size=8 \
        critic_train.parallel.pipeline_parallel_size=1 \
        critic_train.parallel.data_parallel_size=1 \
        critic_inf.parallel.model_parallel_size=4 \
        critic_inf.parallel.pipeline_parallel_size=2 \
        critic_inf.parallel.data_parallel_size=1 \
        ref_inf.parallel.model_parallel_size=4 \
        ref_inf.parallel.pipeline_parallel_size=2 \
        ref_inf.parallel.data_parallel_size=1 \
        rew_inf.parallel.model_parallel_size=1 \
        rew_inf.parallel.pipeline_parallel_size=4 \
        rew_inf.parallel.data_parallel_size=2 