import api.config
import api.data
import impl.data
import impl.model

if __name__ == "__main__":
    dataset_config = api.config.Dataset(
        "wpsf_plrw_packed",
        args=dict(
            contrastive_dim=6,
            enforce_one_or_less_pos=False,
            n_tokens_per_batch=5120,
            max_n_seqs_per_batch=100,
            max_length=1024,
            json_path="/data/aigc/llm/datasets/wps-formula-rw/dataset_val.jsonl",
        ))
    dataloader_cfg = api.config.DataLoader('iterable_dataset_loader')
    dataset = api.data.make_dataset(dataset_config,
                                    seed=1,
                                    ddp_rank=0,
                                    world_size=1,
                                    tokenizer_or_tokenizer_name="/data/aigc/llm/checkpoints/4l-starcoder/",
                                    experiment_name="dataset_test",
                                    trial_name='test')
    dataloder = api.data.make_dataloader(dataloader_cfg, dataset)

    for x in dataloder:
        assert isinstance(x, dict)
