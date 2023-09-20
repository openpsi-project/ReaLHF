import api.config
import api.data
import impl.data
import impl.model

if __name__ == "__main__":
    assert False, "This test is currently corrupted."
    root_dir = "/data"
    model_path = f"{root_dir}/aigc/llm/checkpoints/starcoder-wps-best/"
    max_prompt_len = max_answer_len = 256
    max_seq_len = 512
    contrastive_dim = 5

    dataset_cfg = api.config.Dataset(
        'excel_prompt',
        args=dict(
            dataset_path="/data/aigc/llm/datasets/wps-prompts/train-small.json",
            max_seq_len=max_prompt_len,
        ),
    )
    dataloader_cfg = api.config.DataLoader(
        'excel_rlhf',
        args=dict(
            max_token_len=max_prompt_len,
            shuffle=True,
            drop_last=False,
            batch_size=3,
        ),
    )
    dataset = api.data.make_dataset(dataset_cfg,
                                    seed=1,
                                    ddp_rank=0,
                                    world_size=1,
                                    tokenizer_or_tokenizer_name=model_path,
                                    experiment_name="dataset_test",
                                    trial_name='test')
    dataloder = api.data.make_dataloader(dataloader_cfg, dataset)

    for x in dataloder:
        print(x)
