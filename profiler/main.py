from profiler.experiments import *
import profiler.comm_main

from api.config import _LLM_ENVVARS
import scheduler.client


def profile():
    exp = ProfileExperiment()
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
               f"--batch_size_list {batch_size_list} --seq_len_list {seq_len_list} "\
               f"--use_sequence_parallel"

    for model_path, model_type in zip(exp.model_paths, exp.model_types):
        bs_sl_set = set()
        for rpc in exp.model_rpcs:
            rpc_model_type = exp.model_names_to_types[rpc.model_name]
            if rpc_model_type == model_type:
                bs_sl_set.add((rpc.min_n_seqs, rpc.max_n_tokens // rpc.min_n_seqs))

        bs_list = [bs for bs, _ in bs_sl_set]
        sl_list = [sl for _, sl in bs_sl_set]

        print(f"Profiling {model_type} layers, model path {model_path}, "
              f"cmd {profile_layers_cmd(model_path, model_type, bs_list, sl_list)}")
        sched.submit_array(
            worker_type="profile_layer",
            cmd=profile_layers_cmd(model_path, model_type, bs_list, sl_list),
            count=1,
            cpu=64,
            gpu=8,
            gpu_type="tesla",
            mem=500000,
            env_vars=base_environs,
            container_image="llm/llm-gpu",
        )

    try:
        sched.wait(timeout=None)
    except (KeyboardInterrupt, scheduler.client.JobException, TimeoutError) as e:
        sched.stop_all()
        # raise e

    # print(f"Profiling communication of mesh {exp.device_mesh_name}")
    # profiler.comm_main.main()


def example_main():
    profile()


if __name__ == "__main__":
    example_main()
