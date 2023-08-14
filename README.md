# A Distributed System for LLM RLHF

## Step 1: Train A Reward Model

Command
```
python3 -m apps.main start -e wps-rw-pl-s1 -f ${your_trial_name} --mode slurm --wandb_mode disabled
```

This will train a Plackett-Luce reward model based on give positive and negative examples.

See `ExcelPlackettLuceRewardDataset` in `impl/data/wps_dataset.py` for the used dataset.

See `WPSPlackettLuceRewardInterface` in `impl/model/interface/wps_actor_critic.py` for the algorithm implementation.

The trained model will be saved in the subfolders of `/data/aigc/llm/root/checkpoints/wps-rw-pl-s1/${your_trial_name}/default/epoch4step0/`.

## Step 2: RLHF

Command
```
python3 -m apps.main start -e wps-rlhf -f ${your_trial_name} --wandb_mode disabled --mode slurm
```

Remember to replace the default actor model and reward model to yours.

Check the logs hinted by stdout for detailed training information.

There are some scripts to wash data based on heuristics in `scripts/data`.

## Reward vs Training Steps

![Reward vs Training Steps](assets/reward.png)
