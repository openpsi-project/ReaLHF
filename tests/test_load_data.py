import realhf.api.core.system_api
import realhf.base.namedarray as namedarray
import realhf.impl.dataset
import realhf.impl.model
from realhf.api.core import data_api, system_api
from realhf.base import constants, logging

logger = logging.getLogger("tests.test_load_data")

TOKENIZER_PATH = "/lustre/public/pretrained_model_weights/Llama-2-7b-hf"
PROMPT_ONLY_PATH = "/lustre/meizy/data/antropic-hh/ppo_prompt_only.jsonl"
PROMPT_ANSWER_PATH = "/lustre/fw/datasets/imdb/rl/rm_paired-train-lite.jsonl"


def test_prompt_only(
    max_length: int = 256, pad_to_max_length: bool = False, seed: int = 1
):
    dataset_config = system_api.Dataset(
        "prompt",
        args=dict(
            dataset_path=PROMPT_ONLY_PATH,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
        ),
    )
    dataset = data_api.make_dataset(
        dataset_config,
        seed=seed,
        ddp_rank=0,
        world_size=1,
        tokenizer_or_tokenizer_name=TOKENIZER_PATH,
        experiment_name="dataset_test",
        trial_name="test",
    )
    for i in range(len(dataset)):
        x = dataset[i]
        assert isinstance(x, namedarray.NamedArray)
        assert "seqlens" in x.metadata
        assert isinstance(x.metadata["seqlens"], list)
        for seqlen in x.metadata["seqlens"]:
            assert isinstance(seqlen, int)

    dataloader = data_api.make_dataloader("packed", dataset)

    for i, data in enumerate(dataloader):
        assert isinstance(data, namedarray.NamedArray)
        assert "packed_prompts" in data
        logger.info(f"Batch {i} shape = {data['packed_prompts'].shape}")


def test_prompt_answer(max_length: int = 128, seed: int = 1):
    dataset_config = system_api.Dataset(
        "rw_pair",
        args=dict(
            max_length=max_length,
            dataset_path=PROMPT_ANSWER_PATH,
        ),
    )
    dataset = data_api.make_dataset(
        dataset_config,
        seed=seed,
        ddp_rank=0,
        world_size=1,
        tokenizer_or_tokenizer_name=TOKENIZER_PATH,
        experiment_name="dataset_test",
        trial_name="test",
    )
    for i in range(len(dataset)):
        x = dataset[i]
        assert isinstance(x, namedarray.NamedArray)
        assert "seqlens" in x.metadata
        assert isinstance(x.metadata["seqlens"], list)
        for seqlen in x.metadata["seqlens"]:
            assert isinstance(seqlen, int)

    dataloader = data_api.make_dataloader("packed", dataset)
    for i, x in enumerate(dataloader):
        assert isinstance(x, namedarray.NamedArray)
        assert isinstance(x.metadata["seqlens"], list)
        assert "packed_input_ids" in x
        assert len(x["prompt_lens"]) == len(x["group_factor"])
        logger.info(f"Batch {i} shape = {x['packed_input_ids'].shape}")


if __name__ == "__main__":
    test_prompt_only()
    test_prompt_answer()
