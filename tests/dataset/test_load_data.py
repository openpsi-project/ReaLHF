from reallm.api.core import data_api, system_api
import reallm.api.core.system_api
import reallm.base.namedarray as namedarray
import reallm.impl.dataset
import reallm.impl.model


def test_prompt_answer():
    tpath = "/lustre/public/pretrained_model_weights/Llama-2-7b-hf"
    dpath = "/lustre/fw/datasets/imdb/rl/rm_paired-train-lite.jsonl"
    dataset_config = system_api.Dataset(
        "rw_pair",
        args=dict(
            max_length=1024,
            dataset_path=dpath,
        ),
    )
    dataset = data_api.make_dataset(
        dataset_config,
        seed=1,
        ddp_rank=0,
        world_size=1,
        tokenizer_or_tokenizer_name=tpath,
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

    dataloder = data_api.make_dataloader("packed", dataset)
    for x in dataloder:
        assert isinstance(x, namedarray.NamedArray)
        assert isinstance(x.metadata["seqlens"], list)


if __name__ == "__main__":
    test_prompt_answer()
