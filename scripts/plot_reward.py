import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

experiment_name = "wps-rlhf"
trial_name = "test20230806-5"

reward_lines = subprocess.check_output(
    [f"cat /data/aigc/llm/logs/fw/flash-ppo-s42_run20231101/master_worker-0 | grep task_reward"],
    shell=True).decode('ascii').split('\n')[:-1]
rewards = np.array([float(line.split("'task_reward': ")[-1].split(',')[0]) for line in reward_lines])
print(rewards)
# rewards = np.convolve(rewards, np.ones(10) / 10, mode='same')

plt.plot(rewards)
plt.savefig("reward.png")
