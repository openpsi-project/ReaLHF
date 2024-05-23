from typing import Union
import dataclasses


@dataclasses.dataclass
class PromptAnswerDatasetConfig:
    """Datasets used for SFT.

    The raw data must be a json or jsonl file, where each piece of data is a dictionary
    with keys `prompt` and `answer`. Both `prompt` and `answer` are strings.
    Check `impl/dataset/common/packed_prompt_answer_dataset.py` or inspect the example
    dataset for more details.

    Sequences will be packed into an 1D tensor without padding, so the batch size is determined
    by the number of tokens. The user has the responsibility to control the number of sequences
    in each batch by adjusting `max_seqlen` and `train_tokens_per_batch`, such that it is greater
    than or equal to the data parallelism degree. For example, if `model.parallel.data_parallel_size`
    is set to 4, then the number of sequences in each batch should be at least 4.

    Args:
        train_path (str): Path to the training dataset.
        valid_path (str): Path to the evaluation dataset.
        max_seqlen (str): Maximum sequence length (prompt + answer).
            Sequences longer than this will be truncated.
        train_tokens_per_batch (int): Number of tokens in each batch during training.
        valid_tokens_per_batch (int): Number of tokens in each batch during validation.
    """

    train_path: str = ""
    valid_path: str = ""
    max_seqlen: int = 1024
    train_tokens_per_batch: int = 8192
    valid_tokens_per_batch: int = 8192


@dataclasses.dataclass
class PairedComparisonDatasetConfig:
    """Datasets used for paired-comparison reward modeling and DPO.

    The raw data must be a json or jsonl file, where each piece of data is a dictionary
    with keys `prompt`, `pos_answers`, and `neg_answers`. `prompt` is a string.
    `pos_answers` and `neg_answers` are lists of strings. They must have the same size.
    Check `impl/dataset/common/packed_rw_paired_dataset.py` or inspect the example
    dataset for more details.

    Answer pairs of the same prompt will be sampled in the same batch. Hence, the number of sequences
    in each batch must be even, in the form of [P1A1+, P1A1-, P1A2+, P1A2-, P2A1+, P2A1-, P2A2+, P2A2-, ...],
    where `P` means prompt, `A` means answer, `+` means positive, and `-` means negative.

    Sequences will be packed into an 1D tensor without padding, so the batch size is determined
    by the number of tokens. The user has the responsibility to control the number of *answer pairs*
    in each batch by adjusting `max_seqlen` and `train_tokens_per_batch`, such that it is greater
    than or equal to the data parallelism degree.

    The raw dataset may contain multiple answer pairs for each prompt. We will randomly sample
    `max_pairs_per_prompt` answer pairs for each prompt.

    Args:
        max_pairs_per_prompt (int): Maximum number of answer pairs per prompt.
        max_seqlen (str): Maximum sequence length (prompt + answer).
            Sequences longer than this will be truncated.
        train_tokens_per_batch (int): Number of tokens in each batch during training.
        valid_tokens_per_batch (int): Number of tokens in each batch during evaluation.
        train_path (str): Path to the training dataset.
        valid_path (str): Path to the evaluation dataset.
    """

    train_path: str = ""
    valid_path: str = ""
    max_pairs_per_prompt: int = 2
    max_seqlen: int = 1024
    train_tokens_per_batch: int = 32768
    valid_tokens_per_batch: int = 32768


@dataclasses.dataclass
class PromptOnlyDatasetConfig:
    """Datasets used for PPO RLHF.

    The raw data must be a json or jsonl file, where each piece of data is a dictionary
    with a single key called `prompt`, which is a string.
    Check `impl/dataset/common/prompt_dataset.py` or inspect the example
    dataset for more details.

    Sampled prompts will be left-padded to `max_prompt_len` for generation.

    Args:
        max_prompt_len (int): Maximum prompt length. Prompts shorter than this will be left-padded
            and prompts longer than this will be truncated.
        n_tokens_per_batch (int): Number of tokens in each batch, used for packed dataset
        batch_size (int): Number of prompts in each batch, used for non-packed dataset
        path (str): Path to the dataset.
    """

    max_prompt_len: int = 256
    n_tokens_per_batch: int = 65536
    batch_size: int = 256
    path: str = "/lustre/fw/datasets/imdb/rl/ppo_prompt.jsonl"
    pad_to_max_length: bool = False


DatasetType = Union[PromptOnlyDatasetConfig, PromptAnswerDatasetConfig, PairedComparisonDatasetConfig]
