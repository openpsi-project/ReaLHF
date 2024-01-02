from collections import defaultdict
import multiprocessing as mp
import os
import random
import time

import viztracer

from base.monitor import get_tracer
import base.constants
import base.topology

NUM_PP = 4
NUM_MP = 1
NUM_DP = 1
WORLD_SIZE = NUM_PP * NUM_MP * NUM_DP


def setup(rank):
    os.environ["DLLM_TRACE"] = "1"
    base.constants.set_experiment_trial_names("test", "test")
    base.constants.set_model_name("test_model")
    topo = base.topology.PipeModelDataParallelTopology(num_pp=NUM_PP, num_mp=NUM_MP, num_dp=NUM_DP)
    base.constants.set_fake_grid("test_model", rank, topo)


def rank_print(rank, *args, **kwargs):
    print(f"Rank {rank}: ", *args, **kwargs)


def main(rank):
    setup(rank)

    from impl.model.backend.pipe_engine.dynamic_schedule import (DynamicPipeSchedule, GenerationSchedule,
                                                                 InferenceSchedule, Train1F1BSchedule)
    from impl.model.backend.pipe_engine.schedule_controller import (EngineScheduleClient,
                                                                    EngineScheduleController)

    # tracer = get_tracer(
    #         tracer_entries=int(2e6),
    #         # max_stack_depth=10,
    #         ignore_c_function=False,
    #         ignore_frozen=True,
    #         log_async=True,
    #         # min_duration=20,
    #         output_file=f"/home/meizy/logs/viztracer/trace{rank}.json")
    # tracer.start()

    if rank == 0:
        controller = EngineScheduleController(NUM_PP)
        controller.start()

    client = EngineScheduleClient(stage_id=base.constants.pipe_parallel_rank())

    time.sleep(0.5)
    rank_print(rank, "Setup complete")

    started = False
    train_started = False
    count = 0
    st = time.monotonic()
    instructions = defaultdict(list)
    while True:
        if rank == 0 and not started:
            sched = Train1F1BSchedule(num_micro_batches=NUM_PP * 2, num_stages=NUM_PP)
            controller.issue_schedule(sched, 10)
            started = True

        # if rank == 0 and train_started is False:
        #     sched = Train1F1BSchedule(num_micro_batches=NUM_PP * 2, num_stages=NUM_PP)
        #     controller.issue_schedule(sched, 99)
        #     train_started = True
        sched_id, inst, end = client.poll_instruction()

        if sched_id is not None:
            # fake execute
            rank_print(rank, f"Client polled sched_id: instruction {sched_id}:{inst}")
            # if inst.name in ["ForwardPass", "BackwardPass", "ReduceGrads", "OptimizerStep"]:
            #     instructions[sched_id].append(inst)
            # time.sleep(0.01)

            client.post_result(0)
            rank_print(rank, "Client posted result")
            last_recv_time = time.monotonic()

        count += 1
        if time.monotonic() - st > 10:
            # for sched_id, insts in instructions.items():
            #     rank_print(rank, f"Schedule {sched_id} instructions: {insts}; time elapsed {last_recv_time-st}")
            rank_print(rank, f"time elapsed {last_recv_time - st}")
            if rank == 0:
                controller.stop()
            break


if __name__ == "__main__":

    import base.name_resolve as name_resolve
    import base.names as names
    name_resolve.clear_subtree(names.trial_root(experiment_name="test", trial_name="test"))
    ps = [mp.Process(target=main, args=(rank,)) for rank in range(WORLD_SIZE)]
    for p in ps:
        p.start()

    for p in ps:
        p.join()

    # tracer.save()
