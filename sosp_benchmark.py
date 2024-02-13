import os
import math
import enum

class ModelSize(enum.Enum):
    SMALL = 7
    MEDIUM = 13
    LARGE = 34
    XLARGE = 70


def sweep_model_size(model_size: ModelSize):
    n_gpus = 64 if model_size == ModelSize.XLARGE else 32 if model_size == ModelSize.LARGE else 16
    pp_sizes = [1, 2, 4, 8]
    if model_size == ModelSize.XLARGE:
        pp_sizes.append(16)
    for pp_size in pp_sizes:
        dp_size = n_gpus // pp_size
        exp_name = f"sosp-baseline-a{model_size.value}-{dp_size}x{pp_size}-c7r7"
        os.system(f"python3 -m apps.main start -e {exp_name} -f benchmark")
        if pp_size > 1:
            exp_name = f"sosp-baseline-a{model_size.value}-{dp_size}x{pp_size}-c7r7-mb1"
            os.system(f"python3 -m apps.main start -e {exp_name} -f benchmark")
            exp_name = f"sosp-baseline-a{model_size.value}-{dp_size}x{pp_size}-c7r7-mb1gen"
            os.system(f"python3 -m apps.main start -e {exp_name} -f benchmark")


if __name__ == "__main__":
    # sweep_model_size(ModelSize.SMALL)
    # sweep_model_size(ModelSize.MEDIUM)
    sweep_model_size(ModelSize.LARGE)
    sweep_model_size(ModelSize.XLARGE)
