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
    file_name = "/home/meizy/logs/pipe_mqat.log"
    identifiers = [str(i) for i in range(4)]
    fn = "/workspace/scripts/figs/data_load.png"
    # start_keys = [
    #     "forward_prepare_start", "outer_module_forward_start", "post_process_start", "reserve_kv_cache_start",
    #     "postprocess_cache_start", "genstep_start"
    # ]
    # end_keys = [
    #     "forward_prepare_end", "outer_module_forward_end", "post_process_end", "reserve_kv_cache_end",
    #     "postprocess_cache_end", "genstep_end"
    # ]
    # start_keys = [
    #     "LoadMicroBatch_start", "ForwardPass_start", "LoadNextTokens_start", "SendNextTokens_start",
    #     "RecvNextTokens_start", "RecvActivation_start", "SendActivation_start"
    # ]
    # end_keys = [
    #     "LoadMicroBatch_end", "ForwardPass_end", "LoadNextTokens_end", "SendNextTokens_end",
    #     "RecvNextTokens_end", "RecvActivation_end", "SendActivation_end"
    # ]
    # start_keys = [
    #     "pipe_cache_load_start", "input_tuple_clone_start", "tensor_to_tuple_start", "tuple_to_tensor_start",
    #     "layer_0_start", "layer_1_start", "layer_2_start"
    # ]
    # end_keys = [
    #     "pipe_cache_load_end", "input_tuple_clone_end", "tensor_to_tuple_end", "tuple_to_tensor_end",
    #     "layer_0_end", "layer_1_end", "layer_2_end"
    # ]
    start_keys = ["tensor_to_tuple_start", "tuple_to_tensor_start"]
    end_keys = ["tensor_to_tuple_end", "tuple_to_tensor_end"]
    # start_keys = ["to_tensor_start", "from_tensor_start"]
    # end_keys = ["to_tensor_end", "from_tensor_end"]
    # start_keys = [
    #     f"load_next_tokens_{i}_start" for i in range(1, 5)
    # ]
    # end_keys = [
    #     f"load_next_tokens_{i}_end" for i in range(1, 5)
    # ]
    summary_time_points(
        file_name=file_name,
        start_keys=start_keys,
        end_keys=end_keys,
        identifiers=identifiers,
        step_range=(0, 1000),
        save_fig_path=fn,
    )


def plot_pipeline_rlhf():
    # dir_name = "/lustre/aigc/llm/logs/meizy/wpsf-flash-ppo-pipe_TEST-20231101-01"
    file_name = "/lustre/aigc/llm/logs/meizy/wpsf-flash-ppo-pipe_TEST-20231101-02/model_worker-0"
    identifiers = [str(i) for i in range(4)]
    fn = "scripts/figs/pipeline_rlhf.png"
    start_keys = [
        "LoadMicroBatch_start", "RecvActivation_start", "SendActivation_start", "RecvGrad_start",
        "RecvGrad_end", "ForwardPass_start", "BackwardPass_start", "LoadNextTokens_start",
        "SendNextTokens_start", "RecvNextTokens_start"
    ]
    end_keys = [
        "LoadMicroBatch_end", "RecvActivation_end", "SendActivation_end", "RecvGrad_start", "RecvGrad_end",
        "ForwardPass_end", "BackwardPass_end", "LoadNextTokens_end", "SendNextTokens_end",
        "RecvNextTokens_end"
    ]
    summary_time_points(file_name=file_name,
                        start_keys=start_keys,
                        end_keys=end_keys,
                        identifiers=identifiers,
                        step_range=(0, 2),
                        save_fig_path=fn,
                        figsize=(20, 4))


if __name__ == "__main__":
    # plot_debug_pipeline()
    plot_pipeline_rlhf()
