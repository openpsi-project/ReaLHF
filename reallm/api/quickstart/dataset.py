from typing import Union
import dataclasses


@dataclasses.dataclass
class PromptAnswerDatasetConfig:
    """Datasets used for SFT.

    The raw data must be a json or jsonl file, where each piece of data is a dictionary
    with keys `prompt` and `answer`. Both `prompt` and `answer` are strings.
    Check `impl/dataset/common/packed_prompt_answer_dataset.py` or inspect the example
    dataset for more details.

    Args:
        train_path (str): Path to the training dataset.
        valid_path (str): Path to the evaluation dataset.
        max_seqlen (str): Maximum sequence length (prompt + answer).
            Sequences longer than this will be truncated.
        train_bs_n_seqs (int): Number of sequences in each batch during training.
        valid_bs_n_seqs (int): Number of sequences in each batch during validation.
    """

    train_path: str = ""
    valid_path: str = ""
    max_seqlen: int = 1024
    train_bs_n_seqs: int = 256
    valid_bs_n_seqs: int = 256


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

    The raw dataset may contain multiple answer pairs for each prompt. We will randomly sample
    `max_pairs_per_prompt` answer pairs for each prompt.

    Args:
        train_path (str): Path to the training dataset.
        valid_path (str): Path to the evaluation dataset.
        max_pairs_per_prompt (int): Maximum number of answer pairs per prompt.
        max_seqlen (str): Maximum sequence length (prompt + answer).
            Sequences longer than this will be truncated.
        train_bs_n_seqs (int): Number of sequences in each batch during training.
        valid_bs_n_seqs (int): Number of sequences in each batch during validation.
    """

    train_path: str = ""
    valid_path: str = ""
    max_pairs_per_prompt: int = 2
    max_seqlen: int = 1024
    train_bs_n_seqs: int = 256
    valid_bs_n_seqs: int = 256


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
        train_bs_n_seqs (int): Number of prompts in each batch.
        path (str): Path to the dataset.
    """

    path: str = ""
    max_prompt_len: int = 256
    train_bs_n_seqs: int = 256
    pad_to_max_length: bool = False


DatasetType = Union[PromptOnlyDatasetConfig, PromptAnswerDatasetConfig, PairedComparisonDatasetConfig]
