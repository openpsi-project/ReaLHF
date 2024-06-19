################
Customization
################


Customizing Datasets
-----------------------------------

Overview
~~~~~~~~~~

We provide three types of datasets implementation in ``realhf/impl/dataset/``,
with corresponding configurations

- :class:`realhf.PromptAnswerDatasetConfig`
- :class:`realhf.PairedComparisonDatasetConfig`
- :class:`realhf.PromptOnlyDatasetConfig`.

Please check the corresponding configurations for more details
about how to use or change these implemented datasets.

Datasets in ReaL are the commonly used
`PyTorch map-style datasets <https://pytorch.org/docs/stable/data.html#map-style-datasets>`_.
Users are required to implement a ``__getitem__`` method in the dataset class,
which returns an :class:`realhf.NamedArray` object containing the data of a single sample and its sequence length.
The sequence length is required because ReaL uses variable-length inputs without padding to save GPU memory.

How dataset configuration is parsed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We take the SFT experiment as an example.
The :class:`realhf.PromptAnswerDatasetConfig` class will be converted to a dataset config
under the system API, i.e., ``realhf.api.core.system_api.Dataset``.
Please check the ``datasets`` method of :class:`realhf.SFTConfig` for more details.
This object has a dataset name (in this case, "prompt_answer") and corresponding arguments
that are passed to the dataset class's constructor.

At the end of ``realhf.impl.dataset.prompt_answer_dataset``, we can see a line:

.. code-block:: python

    data_api.register_dataset("prompt_answer", PromptAnswerDataset)

This line properly registers the dataset class with the system API, so that when this name
is given to system API, ReaL can find this dataset implementation and construct it.
The ``args`` field in ``realhf.api.core.system_api.Dataset`` will be passed to the ``__init__``
method of the dataset class, except that ReaL preserves a ``util`` field to store some utility objects.

Steps for implementing a new dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Create a new dataset file under ``realhf/impl/dataset/``.

- Implement a map-style PyTorch dataset class with a ``__getitem__`` method. This method returns an :class:`realhf.NamedArray` object containing the sequence length as metadata.

- Register the class with ``data_api.register_dataset`` at the end of this file, with the name "my-dataset".

- Change the name of the used dataset in experiment configurations, e.g., in the ``datasets`` method of ``realhf.SFTConfig``, to "my-dataset".

- If you would like to pass in more arguments to construct the dataset class, change the quickstart configuration class (in this case, ``realhf.PromptAnswerDatasetConfig``) as well as the ``args`` field in the system API dataset object.


Customizing Models
-----------------------------------

Overview
~~~~~~~~~~

For efficiency reasons, ReaL does not support every transformer model from the HuggingFace model hub.
In ReaL, we implement a :class:`realhf.impl.model.nn.real_llm_api.ReaLModel` class that wraps the HuggingFace model and provides
additional offload and parameter reallocation APIs.


Note that there are some helper functions in the model API that are used to convert HuggingFace models back-and-forth,
e.g., ``from_llama``, ``config_to_codellama``, etc.
These helper functions are generated *automatically* by registering converting functions in the
``api/from_hf/`` folder.

We take ``api/from_hf/llama.py`` as an example.
To register a convertable HuggingFace model, the user should implement\:

- Two functions that convert model configs between HuggingFace and :class:`realhf.ReaLModelConfig`.
- Two functions that convert model state dicts between HuggingFace and ReaL, basically key remap.
- Three functions specifying the names of parameters in the embedding layer, transformer blocks, and the output layer, respectively.

Steps to support a new HuggingFace model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Create a new model file under ``api/from_hf/``.
- Implement the required helper functions as described above.
- Register the model with ``register_hf_family`` at the end of this file.
- (Optional) Test the consistency of the implemented model with scripts in ``tests/``.

We acknowledge that the current config and implementation of ``ReaLModel`` does not support
all the features of HuggingFace models, e.g., MoE, shared embeddings, etc.
As a result, supporting a new HF model usually requires to modify files in ``impl/model/nn/``,
which can be a terrible experience to users that are not familar with the code architecture.
If you have any questions or want to request a new model feature,
please feel free to raise an issue on our GitHub repository.


Customizing Algorithms
------------------------------------

Overview
~~~~~~~~~~~

Algorithms in ReaL are represented as dataflow graphs.
Each node in the graph is a model function call (MFC), which is one of
the generate, inference, or train requests applied to a specific model (e.g., Actor or Critic).
Edges in the graph denote the data or parameter version dependencies
between nodes.

We show the dataflow graph of PPO in the following figure:

.. image:: images/rlhf_dfg.svg
    :alt: Dataflow graph of RLHF.
    :align: center

A node is represented by a :class:`realhf.MFCDef` object.
We can see that the node has a ``model_name`` field and a ``interface_type`` field,
which specifies what this node should conceptually do during exection.
The ``interface_impl`` field specifies an actual implementation of the model interface.

The interface class has the following signature:

.. autoclass:: realhf.ModelInterface
    :members:
    :undoc-members:

During the execution of an MFC node, the model with ``model_name`` will be passed
into this interface object together with the data specified in the MFC node.

.. note::
    Similar to datasets, model interfaces are also registered and constructed by the system API.
    Please check ``impl/model/interface/sft_interface.py`` for an example.
    The ``SFTInterface`` is registered at the end of this file and constructed by :class:`realhf.SFTConfig`
    (see the ``rpcs`` method).

Running algorithms in ReaL is exactly running a large dataflow graph that
concatenates all the training iterations.
The *MasterWorker* monitors the running state of this graph and issues MFC requests
to *ModelWorkers* once the dependencies are satisfied.
For more details about the code architecture, please refer to the :doc:`arch` page.

.. To implement a new algorithm in ReaL,
.. the user should first figure out whether the new dataflow can be unified into
.. existing dataflow graphs (i.e., SFT/RW, DPO, PPO), as defined in the ``experiments/common/`` folder.
.. If the new algorithm has a completely new dataflow, the user should modify
.. the experiment configuration class.
.. Otherwise, e.g., if the user just want to add an additional loss term to the existing algorithm,
.. the user can just modify the interface implementation in the ``impl/model/interface`` folder.

Example 1: Replace the interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Say, in the PPO experiment, we want to use a customized reward model
from HuggingFace. How should we do if this model is not supported by ``ReaLModel``?

We provide the example code in ``examples/ppo_sentiment.py``, where we replace the
trained reward model for sentiment generation with a BERT-like sentiment analysis model
from HuggingFace.

First, we should implement a new model interface class for our customized usage:

.. code-block:: python

    @dataclasses.dataclass
    class SentimentScoringInterface(model_api.ModelInterface):

        def __post_init__(self):
            super().__post_init__()
            self.score_model = (
                transformers.AutoModelForSequenceClassification.from_pretrained(
                    "/path/to/score_model"
                ).cuda()
            )
            self.score_model.eval()
            self.score_tokenizer = transformers.AutoTokenizer.from_pretrained(
                "/path/to/score_model"
            )

        @torch.no_grad()
        def inference(self, model: model_api.Model, data: NamedArray) -> NamedArray:
            ...
            # Re-tokenize.
            texts = model.tokenizer.batch_decode(
                input_ids, skip_special_tokens=True
            )
            encoding = self.score_tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            )

            # Inference to get the score.
            # For IMDB, 0 is negative and 1 is positive. We record the logits of positive.
            scores = self.score_model(
                input_ids=encoding["input_ids"].cuda(),
                attention_mask=encoding["attention_mask"].cuda(),
            ).logits[..., -1].contiguous().float()
            scores = logits[..., -1].contiguous().float()

            res = NamedArray(scores=scores)
            res.register_metadata(**data.metadata)
            return res

Here are two key points in this code:

- During interface initialization, we load a HuggingFace model and its tokenizer.
- During inference, we re-tokenize the generated output from the Actor, compute the score, and return it.

That's easy, right? Now we should register this interface in the system API:

.. code-block:: python

    model_api.register_interface("sentiment_scoring", SentimentScoringInterface)

Then, to use our customized interface implementation in PPO, we should change
the ``interface_impl`` field of the reward model in the MFC nodes of PPO:

.. code-block:: python

    class MyPPOConfig(PPOConfig):

        def initial_setup(self) -> ExperimentConfig:
            ...
            for mw in cfg.model_worker:
                for s in mw.shards:
                    if s.id.model_name.role == "reward":
                        # Remove the original reward model because we use the customized one.
                        s.model = config_api.Model(
                            "tokenizer",
                            args=dict(
                                tokenizer_path=self.rew.path,
                            ),
                        )
                        s.backend = config_api.ModelBackend("null")
            # Change the MFC of Reward.
            idx = 0
            for rpc in cfg.model_rpcs:
                if rpc.model_name.role == "reward":
                    break
                idx += 1
            inf_reward_rpc = cfg.model_rpcs[idx]
            inf_reward_rpc.interface_impl = dfg.ModelInterface("sentiment_scoring")
            inf_reward_rpc.post_hooks = []
            return cfg

Don't forget the register your customized experiment configuration
such that ReaL can launch it with the quickstart command line options:

.. code-block:: python

    register_quickstart_exp("my-ppo", MyPPOConfig)

Done! Let's run the customized experiment with the quickstart command:

.. code-block:: console

    # Note that we change the name "ppo" to "my-ppo"
    python3 -m realhf.apps.quickstart my-ppo \
        experiment_name=sentiment-ppo \
        trial_name=test \
        ppo.kl_ctl=0.1 \
        ppo.top_p=0.9 ppo.top_k=1000 \
        ...

This example also applies for scenarios when you want to use an external reward,
like the signal from compiler or other online automatic evaluations.

Example 2: Develop a new dataflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO