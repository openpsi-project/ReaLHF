import os
import sys

from base.monitor import summary_time_points


def plot_pipeline():
    dir_name = "/lustre/aigc/llm/logs/meizy/wpsf-sft-flash-pipe-s1_TEST-PLOT-20231023-03"
    identifiers = [str(i) for i in range(8)]
    # st = 1698044544 * 10e9
    # et = 1698044550 * 10e9
    st = 1698044545235232600
    et = 1698044550861552000
    fn = "/home/meizy/distributed_llm/scripts/figs/pp_8gpus_starcoder.png"
    start_keys = [
        "LoadMicroBatch_start", "RecvActivation_start", "SendActivation_start", "RecvGrad_start",
        "RecvGrad_end", "ForwardPass_start", "BackwardPass_start"
    ]
    end_keys = [
        "LoadMicroBatch_end", "RecvActivation_end", "SendActivation_end", "RecvGrad_start", "RecvGrad_end",
        "ForwardPass_end", "BackwardPass_end"
    ]
    summary_time_points(
        dir_name=dir_name,
        start_keys=start_keys,
        end_keys=end_keys,
        identifiers=identifiers,
        start_time=st,
        end_time=et,
        save_fig_path=fn,
    )


def plot_rlhf():
    dir_name = "/lustre/aigc/llm/logs/meizy/chat-rlhf-benchmark_TEST-20231023-02"
    identifiers = ["actor_0", "actor_1", "critic_0", "critic_1", "ref_0", "reward_0"]
    st = 169805003320
    et = 169805003800
    fn = "/home/meizy/distributed_llm/scripts/figs/rlhf.png"
    start_keys = [
        "actor_generate_start", "actor_train_start", "critic_inference_start", "critic_train_start",
        "ref_inference_start", "reward_inference_start"
    ]
    end_keys = [
        "actor_generate_end", "actor_train_end", "critic_inference_end", "critic_train_end",
        "ref_inference_end", "reward_inference_end"
    ]
    summary_time_points(
        dir_name=dir_name,
        start_keys=start_keys,
        end_keys=end_keys,
        identifiers=identifiers,
        start_time=st,
        end_time=et,
        save_fig_path=fn,
    )


def plot_debug_pipeline():
    file_name = "/home/meizy/logs/new_train_time.log"
    identifiers = [str(i) for i in range(4)]
    fn = "/workspace/scripts/figs/new_train_time.png"
    start_keys = [
        "LoadMicroBatch_start", "RecvActivation_start", "SendActivation_start", "RecvGrad_start",
        "RecvGrad_end", "ForwardPass_start", "BackwardPass_start", "LoadNextTokens_start",
        "SendNextTokens_start", "RecvNextTokens_start", "OptimizerStep_start", "Prepare_start"
    ]
    end_keys = [
        "LoadMicroBatch_end", "RecvActivation_end", "SendActivation_end", "RecvGrad_start", "RecvGrad_end",
        "ForwardPass_end", "BackwardPass_end", "LoadNextTokens_end", "SendNextTokens_end",
        "RecvNextTokens_end", "OptimizerStep_end", "Prepare_end"
    ]
    summary_time_points(file_name=file_name,
                        start_keys=start_keys,
                        end_keys=end_keys,
                        identifiers=identifiers,
                        step_range=(1, 2),
                        save_fig_path=fn,
                        figsize=(20, 4))


def plot_pipeline_rlhf():
    # dir_name = "/lustre/aigc/llm/logs/meizy/wpsf-flash-ppo-pipe_TEST-20231101-01"
    file_name = "/lustre/aigc/llm/logs/meizy/wpsf-flash-ppo-pipe_TEST-20231101-02/model_worker-0"
    identifiers = [str(i) for i in range(4)]
    fn = "scripts/figs/pipeline_rlhf.png"
    start_keys = [
        "LoadMicroBatch_start", "RecvActivation_start", "SendActivation_start", "RecvGrad_start",
        "RecvGrad_end", "ForwardPass_start", "BackwardPass_start", "LoadNextTokens_start",
        "SendNextTokens_start", "RecvNextTokens_start", "OptimizerStep_start"
    ]
    end_keys = [
        "LoadMicroBatch_end", "RecvActivation_end", "SendActivation_end", "RecvGrad_start", "RecvGrad_end",
        "ForwardPass_end", "BackwardPass_end", "LoadNextTokens_end", "SendNextTokens_end",
        "RecvNextTokens_end", "OptimizerStep_end"
    ]
    summary_time_points(file_name=file_name,
                        start_keys=start_keys,
                        end_keys=end_keys,
                        identifiers=identifiers,
                        step_range=(0, 2),
                        save_fig_path=fn,
                        figsize=(20, 4))


if __name__ == "__main__":
    plot_debug_pipeline()
    # plot_pipeline_rlhf()
