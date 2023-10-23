import os
import sys
sys.path.append("../")

from base.monitor import plot_time_points

def plot_pipeline():
    dir_name = "/lustre/aigc/llm/logs/meizy/wpsf-sft-flash-pipe-s1_TEST-PLOT-20231023-03"
    identifiers = [str(i) for i in range(8)]
    # st = 1698044544 * 10e9
    # et = 1698044550 * 10e9
    st = 1698044545235232600
    et = 1698044550861552000
    fn = "/home/meizy/distributed_llm/scripts/figs/pp_8gpus_starcoder.png"
    start_keys = ["LoadMicroBatch_start", "RecvActivation_start", "SendActivation_start", "RecvGrad_start", "RecvGrad_end", 
                  "ForwardPass_start", "BackwardPass_start"]
    end_keys = ["LoadMicroBatch_end", "RecvActivation_end", "SendActivation_end", "RecvGrad_start", "RecvGrad_end",
                "ForwardPass_end", "BackwardPass_end"]
    plot_time_points(
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
    start_keys = ["actor_generate_start", "actor_train_start", "critic_inference_start", "critic_train_start",
                  "ref_inference_start", "reward_inference_start"]
    end_keys = ["actor_generate_end", "actor_train_end", "critic_inference_end", "critic_train_end",
                "ref_inference_end", "reward_inference_end"]
    plot_time_points(
        dir_name=dir_name,
        start_keys=start_keys,
        end_keys=end_keys,
        identifiers=identifiers,
        start_time=st,
        end_time=et,
        save_fig_path=fn,
    )


if __name__ == "__main__":
    plot_rlhf()