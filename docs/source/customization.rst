################
Customization
################


Customizing Datasets
-----------------------------------

Overview
~~~~~~~~~~

We provide three types of datasets implementation in ``reallm/impl/dataset/``,
with corresponding configurations

- :class:`reallm.PromptAnswerDatasetConfig`
- :class:`reallm.PairedComparisonDatasetConfig`
- :class:`reallm.PromptOnlyDatasetConfig`.

Please check the corresponding configurations for more details
about how to use or change these implemented datasets.

Datasets in ReaL are the commonly used
`PyTorch map-style datasets <https://pytorch.org/docs/stable/data.html#map-style-datasets>`_.
Users are required to implement a ``__getitem__`` method in the dataset class,
which returns an ``NamedArray`` object containing the data of a single sample and its sequence length.
The sequence length is required because ReaL uses variable-length inputs without padding to save GPU memory.

How dataset configuration is parsed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We take the SFT experiment as an example.
The :class:`reallm.PromptAnswerDatasetConfig` class will be converted to a dataset config
under the system API, i.e., ``reallm.api.core.system_api.Dataset``.
Please check the ``datasets`` method of :class:`reallm.SFTConfig` for more details.
This object has a dataset name (in this case, "prompt_answer") and corresponding arguments
that are passed to the dataset class constructor.

At the end of ``reallm.impl.dataset.prompt_answer_dataset``, we can see a line:

.. code-block:: python

    data_api.register_dataset("prompt_answer", PromptAnswerDataset)

This line properly registers the dataset class with the system API, so that when this name
is given to system API, ReaL can find this dataset implementation and construct it.
The ``args`` field in ``reallm.api.core.system_api.Dataset`` will be passed to the ``__init__``
method of the dataset class, except that ReaL preserves a ``util`` field to store some utility objects.

Steps for implementing a new dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Create a new dataset file under ``reallm/impl/dataset/``.

- Implement a map-style PyTorch dataset class with a ``__getitem__`` method. This method returns an ``NamedArray`` object containing the sequence length as metadata.

- Register the class with ``data_api.register_dataset`` at the end of this file, with the name "my-dataset".

- Change the name of the used dataset in experiment configurations, e.g., in the ``datasets`` method of ``reallm.SFTConfig``, to "my-dataset".

- If you would like to pass in more arguments to construct the dataset class, change the quickstart configuration class (in this case, ``reallm.PromptAnswerDatasetConfig``) as well as the ``args`` field in the system API dataset object.

Reference Manual for ``NamedArray``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: reallm.base.namedarray.NamedArray
    :members:

.. autofunction::reallm.base.namedarray.from_dict

.. autofunction::reallm.base.namedarray.recursive_aggregate

.. autofunction::reallm.base.namedarray.recursive_apply


Customizing Models
-----------------------------------

Overview
~~~~~~~~~~

For efficiency reasons, ReaL does not support every transformer model from the HuggingFace model hub.
In ReaL, we implement a :class:`ReaLModel` class that wraps the HuggingFace model and provides
additional offload and parameter reallocation APIs.

We provide reference manuals of related classes below.

.. autoclass:: reallm.ReaLModelConfig

.. autoclass:: reallm.impl.model.nn.real_llm_api.ReaLModel
    :members:
    :undoc-members:
    :exclude-members: forward, state_dict, load_state_dict, build_reparallelization_plan, build_reparallelized_layers_async, patch_reparallelization, pre_process, post_process, share_embeddings_and_output_weights

Note that there are some helper functions in the model API that are used to convert HuggingFace models back-and-forth,
e.g., ``from_llama``, ``config_to_codellama``, etc.
These helper functions are generated *automatically* by registering converting functions in the
``api/from_hf/`` folder.

We take ``api/from_hf/llama.py`` as an example.
To register a convertable HuggingFace model, the user should implement\:

- Two functions that convert model configs between HuggingFace and :class:`reallm.ReaLModelConfig`.
- Two functions that convert model state dicts between HuggingFace and ReaL, basically key remap.
- Three functions specifying the names of parameters in the embedding layer, transformer blocks, and the output layer, respectively.

Steps to support a new HuggingFace model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

TODO