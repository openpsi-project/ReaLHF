import os
import subprocess

exp_name = "wps-rw-pl-s1"
trial_name_prefix = "lora-sweep-20230731"
log_dir_path = "/home/aigc/llm/fw/logs/"

count = 0
for root, dirs, files in os.walk(log_dir_path):
    for name in dirs:
        if name.startswith(f"{exp_name}_{trial_name_prefix}"):
            count += 1

for trial in range(max(1, count), 5001):
    os.system(f"python3 -m apps.main start -e {exp_name} -f {trial_name_prefix}-{trial} "
              f"--wandb_mode disabled --mode slurm --debug")
