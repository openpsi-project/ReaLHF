import os
import subprocess

exp_name = "wps-rw-s1"
for trial in range(1, 5001):
    os.system(f"python3 -m apps.main start -e {exp_name} -f lora-sweep-20230729-{trial} "
              f"--wandb_mode disabled --mode slurm --debug")
