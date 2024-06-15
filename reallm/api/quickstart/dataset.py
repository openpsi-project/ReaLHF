from typing import Union
import dataclasses


@dataclasses.dataclass
class PromptAnswerDatasetConfig:
    """Datasets used for SFT.

    The raw data must be a json or jsonl file, where each piece of data is a dictionary
    with keys `prompt` and `answer`. Both `prompt` and `answer` are strings.

    :param train_path: Path to the training dataset.
    :type train_path: str
    :param valid_path: Path to the evaluation dataset.
    :type valid_path: str
    :param max_seqlen: Maximum sequence length (prompt + answer).
        Sequences longer than this will be truncated.
    :type max_seqlen: int
    :param train_bs_n_seqs: Number of sequences in each batch during training.
    :type train_bs_n_seqs: int
    :param valid_bs_n_seqs: Number of sequences in each batch during validation.
    :type valid_bs_n_seqs: int
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

    Answer pairs of the same prompt will be sampled in the same batch. Hence, the number of sequences
    in each batch must be even, in the form of [P1A1+, P1A1-, P1A2+, P1A2-, P2A1+, P2A1-, P2A2+, P2A2-, ...],
    where `P` means prompt, `A` means answer, `+` means positive, and `-` means negative.

    The raw dataset may contain multiple answer pairs for each prompt. We will randomly sample
    `max_pairs_per_prompt` answer pairs for each prompt in each epoch.

    :param train_path: Path to the training dataset.
    :type train_path: str
    :param valid_path: Path to the evaluation dataset.
    :type valid_path: str
    :param max_pairs_per_prompt: Maximum number of answer pairs per prompt.
    :type max_pairs_per_prompt: int
    :param max_seqlen: Maximum sequence length (prompt + answer).
        Sequences longer than this will be truncated.
    :type max_seqlen: int
    :param train_bs_n_seqs: Number of sequences in each batch during training.
    :type train_bs_n_seqs: int
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

    :param path: Path to the training dataset.
    :type path: str
    :param max_prompt_len: Maximum prompt length.
        Prompts longer than this will be truncated.
    :type max_prompt_len: int
    :param train_bs_n_seqs: Number of prompts in each batch.
    :type train_bs_n_seqs: int
    :param pad_to_max_length: Whether to pad prompts to the maximum length.
    :type pad_to_max_length: bool
    """

    path: str = ""
    max_prompt_len: int = 256
    train_bs_n_seqs: int = 256
    pad_to_max_length: bool = False
