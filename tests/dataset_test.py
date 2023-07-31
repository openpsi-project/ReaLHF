import api.config
import api.data
import impl.data
import impl.model

if __name__ == "__main__":
    root_dir = "/home"
    model_path = f"{root_dir}/aigc/llm/checkpoints/starcoder-wps-best/"
    max_seq_len = 512
    contrastive_dim = 5

    dataset_cfg = api.config.Dataset(
        'wps_reward_plackett_luce',
        args=dict(
            dataset_path=f"{root_dir}/aigc/llm/datasets/rw-contrastive/train.jsonl",
            tokenizer_name_or_path=model_path,
            max_seq_len=max_seq_len,
            contrastive_dim=contrastive_dim,
        ),
    )
    dataloader_cfg = api.config.DataLoader(
        'default',
        args=dict(
            shuffle=True,
            drop_last=False,
            batch_size=3,
        ),
    )
    dataset = api.data.make_dataset(dataset_cfg,
                                    seed=1,
                                    ddp_rank=0,
                                    world_size=1,
                                    experiment_name="dataset_test",
                                    trial_name='test',
                                    is_eval=False)
    dataloder = api.data.make_dataloader(dataloader_cfg, dataset)

    for x in dataloder:
        print(x)
