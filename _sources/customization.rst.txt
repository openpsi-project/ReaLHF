################
Customization
################

Customizing Datasets
-----------------------------------

Overview
~~~~~~~~~~

We provide three types of dataset implementations in ``realhf/impl/dataset/`` with the following configurations:

- :class:`realhf.PromptAnswerDatasetConfig`
- :class:`realhf.PairedComparisonDatasetConfig`
- :class:`realhf.PromptOnlyDatasetConfig`

Please refer to the respective configuration documentation for detailed instructions on how to use or modify these datasets.

Datasets in ReaL are commonly used
`PyTorch map-style datasets <https://pytorch.org/docs/stable/data.html#map-style-datasets>`_.
Users need to implement a ``__getitem__`` method in the dataset class,
which returns a :class:`realhf.NamedArray` object containing the data of a single sample and its sequence length.
The sequence length is necessary because ReaL uses variable-length inputs without padding to save GPU memory.

How Dataset Configuration is Parsed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will use the SFT experiment as an example.

The :class:`realhf.PromptAnswerDatasetConfig` object will be converted to a dataset configuration
under the system API, specifically ``realhf.api.core.system_api.Dataset``.
Refer to the ``datasets`` method of :class:`realhf.SFTConfig` for more details.
This object includes a dataset name (in this case, "prompt_answer") and corresponding arguments
that are passed to the dataset class's constructor:

.. code-block:: python

    @property
    def datasets(self):
        return [
            Dataset(
                "prompt_answer",
                args=dict(
                    max_length=self.dataset.max_seqlen,
                    dataset_path=self.dataset.train_path,
                ),
            )
        ]

At the end of ``realhf.impl.dataset.prompt_answer_dataset``, we find the following line:

.. code-block:: python

    data_api.register_dataset("prompt_answer", PromptAnswerDataset)

This line registers the dataset class with the system API. When this name is provided to the system API,
ReaL can locate this dataset implementation and construct it.
The ``args`` field in ``realhf.api.core.system_api.Dataset`` will be passed to the ``__init__``
method of the dataset class, except that ReaL reserves a ``util`` field to store some utility objects.

Steps for Implementing a New Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create a new dataset file under ``realhf/impl/dataset/``.

2. Implement a map-style PyTorch dataset class with a ``__getitem__`` method. This method should return a :class:`realhf.NamedArray` object containing the sequence length as metadata.

3. Register the class with ``data_api.register_dataset`` at the end of the file, using the name "my-dataset".

4. Update the name of the dataset in experiment configurations, for example, in the ``datasets`` method of ``realhf.SFTConfig``, to "my-dataset".

5. If you need to pass additional arguments to construct the dataset class, modify the quickstart configuration class (in this case, ``realhf.PromptAnswerDatasetConfig``) as well as the ``args`` field in the system API dataset object.


Customizing Models
-----------------------------------

Overview
~~~~~~~~~~

For efficiency reasons, ReaL does not support every transformer
model from the HuggingFace model hub.
In ReaL, we implement the :class:`realhf.impl.model.nn.real_llm_api.ReaLModel`
class that wraps the HuggingFace model and provides micro-batched pipelining,
offload, and parameter reallocation functionalities.

There are helper functions in the model API used to convert HuggingFace models back and forth,
such as ``from_llama`` and ``to_llama``. 
These helper functions are generated automatically by registering conversion functions in the ``api/from_hf/`` folder.

For example, consider ``api/from_hf/llama.py``.
To register a convertible HuggingFace model, the user should implement:

- Two functions to convert model configs between HuggingFace and :class:`realhf.ReaLModelConfig`.
- Two functions to convert model state dicts between HuggingFace and ReaL, primarily involving key remapping.
- Three functions specifying the names of parameters in the embedding layer, transformer blocks, and the output layer, respectively.

Steps to Support a New HuggingFace Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Create a new model file under ``api/from_hf/``.
- Implement the required helper functions as described above.
- Register the model with register_hf_family at the end of the file.
- (Optional) Test the consistency of the implemented model with scripts in ``tests/``.

We acknowledge that the current configuration and implementation of ``ReaLModel``
do not support all features of HuggingFace models,
such as MoE.
As a result, supporting a new HuggingFace model
often requires modifications to files in ``impl/model/nn/``,
which can be a challenging experience for users unfamiliar with the code architecture.
If you have any questions or wish to request a new model feature,
please feel free to raise an issue on our GitHub repository.


Customizing Algorithms
------------------------------------

Overview
~~~~~~~~~~~

In ReaL, algorithms are represented as dataflow graphs.
Each node in the graph corresponds to a model function call (MFC),
which can be a generate, inference, or train request applied to a
specific model (e.g., Actor or Critic).
The edges in the graph indicate data or
parameter version dependencies between nodes.

The following figure illustrates the dataflow graph of PPO:

.. image:: images/rlhf_dfg.svg
    :alt: Dataflow graph of RLHF.
    :align: center

A node is represented by a :class:`realhf.MFCDef` object.
Each node has a ``model_name`` field and an ``interface_type``
field, which specify what the node should conceptually do during execution.
The ``interface_impl`` field specifies the actual implementation of the model interface.

The interface class has the following signature:

.. autoclass:: realhf.ModelInterface
    :members:
    :undoc-members:

During the execution of an MFC node,
the model identified by ``model_name`` is passed into this interface object,
along with the data specified in the MFC node.

.. note::

    Similar to datasets, model interfaces are registered and constructed by the system API. Please check ``impl/model/interface/sft_interface.py`` for an example. The ``SFTInterface`` is registered at the end of this file and constructed by :class:`realhf.SFTConfig` (see the ``rpcs`` method).

Running algorithms in ReaL involves executing a large dataflow
graph that concatenates all the training iterations.
The *MasterWorker* monitors the state of this graph and
issues MFC requests to *ModelWorkers* once the dependencies are satisfied.


.. For more details about the code architecture, please refer to the :doc:`arch` page.

Example: A Customized Reward Function for PPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


In this example, we demonstrate how to use a customized reward model from HuggingFace in a PPO experiment when the model is not supported by ``ReaLModel``.

The example code can be found in ``examples/ppo_sentiment.py``, where we replace the trained reward model for sentiment generation with a BERT-like sentiment analysis model from HuggingFace.

First, we need to implement a new model interface class for our customized use:

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
            # Re-tokenize the texts.
            texts = model.tokenizer.batch_decode(
                input_ids, skip_special_tokens=True
            )
            encoding = self.score_tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            )

            # Perform inference to get the score.
            # For IMDB, 0 is negative and 1 is positive. We record the logits of the positive class.
            scores = self.score_model(
                input_ids=encoding["input_ids"].cuda(),
                attention_mask=encoding["attention_mask"].cuda(),
            ).logits[..., -1].contiguous().float()

            res = NamedArray(scores=scores)
            res.register_metadata(**data.metadata)
            return res

Key points in this code:

- During interface initialization, we load a HuggingFace model and its tokenizer.
- During inference, we re-tokenize the generated output from the Actor, compute the score, and return it.

Now, we need to register this interface in the system API:

.. code-block:: python

    model_api.register_interface("sentiment_scoring", SentimentScoringInterface)

To use our customized interface implementation in PPO, we need to change the ``interface_impl`` field of the reward model in the MFC nodes of PPO:

.. code-block:: python

    class MyPPOConfig(PPOConfig):

        def initial_setup(self) -> ExperimentConfig:
            ...
            for mw in cfg.model_worker:
                for s in mw.shards:
                    if s.id.model_name.role == "reward":
                        # Remove the original reward model because we are using a customized one.
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

Don't forget to register your customized experiment configuration so that ReaL can launch it with the quickstart command line options:

.. code-block:: python

    register_quickstart_exp("my-ppo", MyPPOConfig)

Finally, let's run the customized experiment with the quickstart command:

.. code-block:: console

    # Note that we change the name "ppo" to "my-ppo"
    python3 -m realhf.apps.quickstart my-ppo \
        experiment_name=sentiment-ppo \
        trial_name=test \
        ppo.kl_ctl=0.1 \
        ppo.top_p=0.9 ppo.top_k=1000 \
        ...

This example is also applicable for scenarios where you want to use an external reward, such as a signal from a compiler or other online automatic evaluations.


.. Example 2: Develop a new dataflow
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. TODO