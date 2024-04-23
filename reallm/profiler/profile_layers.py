from typing import List
import argparse
import math

from profiler.experiments import *
from profiler.utils import find_factors
import profiler.multi_host_main

from api.config.config_base import MODEL_TYPE_TO_PATH
from api.config.config_system import _LLM_ENVVARS
from api.config.dfg import ModelRPC
import api.config.config_system as config_package
import scheduler.client


def profile_model_type(model_type: ModelType):
    if model_type.is_critic:
        return

    model_path = MODEL_TYPE_TO_PATH[model_type]

    base_environs = {
        "PYTHONPATH": os.path.dirname(os.path.dirname(__file__)),
        "WANDB_MODE": "disabled",
        "DLLM_MODE": "SLURM",
        "DLLM_TRACE": "0",
        **_LLM_ENVVARS,
    }
    sched = scheduler.client.make(mode="slurm", expr_name="profile", trial_name="profile")

    def profile_layers_cmd(model_path, model_type, batch_size_list, seq_len_list):
        batch_size_list = ",".join(str(x) for x in batch_size_list)
        seq_len_list = ",".join(str(x) for x in seq_len_list)
        return f"python -m profiler.layers_main "\
               f"--model_path {model_path} --model_name {model_type} "\
               f"--batch_size_list {batch_size_list} --seq_len_list {seq_len_list}"

    bs_list = [1, 2, 4, 8, 16, 32, 64, 128] * 3
    sl_list = [128] * 8 + [256] * 8 + [512] * 8

    print(f"Profiling {model_type} layers, model path {model_path}, "
          f"cmd {profile_layers_cmd(model_path, model_type, bs_list, sl_list)}")
    sched.submit_array(
        worker_type="profile_layer",
        cmd=profile_layers_cmd(model_path, str(model_type), bs_list, sl_list),
        count=1,
        cpu=64,
        gpu=8,
        gpu_type="tesla",
        nodelist="QH-com29",
        mem=500000,
        env_vars=base_environs,
        container_image="llm/llm-gpu",
    )

    try:
        sched.wait(timeout=None)
    except (KeyboardInterrupt, scheduler.client.JobException, TimeoutError) as e:
        sched.stop_all()
        # raise e


def profile_rpcs(rpcs: List[ModelRPC]):
    model_types = set(rpc.model_type for rpc in rpcs)
    for model_type in model_types:
        profile_model_type(model_type)


def profile(args):
    # trial_name = args.trial_name
    expr_name = args.expr_name
    exp: ProfileExperiment = config_package.make_experiment(expr_name)
    # exp.profile_communication = False
    # exp.profile_rpc = True
    # device_mesh_size = exp.n_nodes * 8
    # find factors of device mesh size
    # factors = find_factors(device_mesh_size)  # possible num_dp and num_pp

    model_type = exp.model_type
    profile_model_type(model_type)


def profile_all_model_types():
    sizes = [7, 13, 34, 70]
    for size in sizes:
        _class = "llama" if size != 34 else "codellama"
        model_type = ModelType(_class=_class, size=size, is_critic=False)
        profile_model_type(model_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a profiling experiment.")
    parser.add_argument(
        "-e",
        "--expr_name",
        type=str,
        default=None,
    )
    # parser.add_argument(
    #     "-f",
    #     "--trial_name",
    #     type=str,
    #     default=None,
    # )
    args = parser.parse_args()

    if args.expr_name is None:
        profile_all_model_types()
    else:
        profile(args)