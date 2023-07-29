import os
import subprocess

lines = [
    '#!/bin/bash',
    f'#SBATCH --job-name=test',
    f'#SBATCH --ntasks=16',
    f'#SBATCH --gpus-per-task=1',
    f'#SBATCH --cpus-per-task=1',
    f'#SBATCH --mem-per-cpu=50000M',
]

srun_env = os.environ.copy()
# Setup step command.
srun_flags = [
    f"--ntasks=16",
    f"--cpus-per-task=1",
    f"--gpus-per-task=1",
    f"--mem-per-cpu=50000M",
    f"--container-image=root/llm-gpu",
    f"--container-mounts=/home:/home",
    "--container-workdir=/home/fw/distributed_llm-wei-dev",
]

srun_cmd = f'srun -l {" ".join(srun_flags)} python3 hello_slurm.py'

lines += [
    srun_cmd,
]
script = '\n'.join(lines).encode('ascii')
r = subprocess.check_output(['sbatch', '--parsable'], input=script, env=srun_env).decode('ascii').strip()
