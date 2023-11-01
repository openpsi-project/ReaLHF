import logging
import time

from base.monitor import mock_time_mark_ms, plot_time_points

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"
logging.basicConfig(filename="/home/meizy/logs/simulated.log",
                    filemode="w",
                    format=LOG_FORMAT,
                    datefmt=DATE_FORMAT,
                    level="DEBUG")

INST_TIME_MAPPING = {"first_generate": 20, "other_generate": 4, "forward": 40, "backward": 80, "optim": 10}


def exec_inst(inst, rank, now):
    t = INST_TIME_MAPPING[inst]
    mock_time_mark_ms(f"{inst}_start", rank, now + 1, step=0)
    mock_time_mark_ms(f"{inst}_end", rank, now + t + 1, step=0)
    return now + t + 1


def wait_for(inst, num, now):
    return now + num * (INST_TIME_MAPPING[inst] + 1)


def wait_for_time(t, now):
    return now + t


def generate_sched(num_pp=4, num_micro_batch=4, max_tokens=5):
    assert num_micro_batch >= num_pp
    for rank in range(num_pp):
        now = 0
        now = wait_for("first_generate", rank, now)
        now = exec_inst("first_generate", rank, now)
        for i in range(max_tokens * num_micro_batch - 1):
            now = exec_inst("other_generate", rank, now)


def generate_sched(num_pp=4, num_micro_batch=4, max_tokens=5):
    assert num_micro_batch >= num_pp
    for rank in range(num_pp):
        now = 0
        now = wait_for("first_generate", rank, now)
        now = exec_inst("first_generate", rank, now)
        for i in range(num_micro_batch - 1):
            now = exec_inst("other_generate", rank, now)

        now = wait_for("first_generate", num_pp, 0)
        now = wait_for("other_generate", rank, now)
        for i in range((max_tokens - 1) * (num_micro_batch - 1)):
            now = exec_inst("other_generate", rank, now)


def plot_simulated_pipeline():
    file_name = "/home/meizy/logs/simulated.log"
    identifiers = [str(i) for i in range(4)]
    fn = "/workspace/scripts/figs/simulated.png"
    start_keys = [
        "first_generate_start", "other_generate_start", "forward_start", "backward_start", "optim_start"
    ]
    end_keys = ["first_generate_end", "other_generate_end", "forward_end", "backward_end", "optim_end"]
    plot_time_points(
        file_name=file_name,
        start_keys=start_keys,
        end_keys=end_keys,
        identifiers=identifiers,
        step_range=None,
        save_fig_path=fn,
    )


if __name__ == "__main__":
    generate_sched(max_tokens=20)
    plot_simulated_pipeline()
