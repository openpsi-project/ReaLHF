import sys

sys.path.append("../")

from impl.model.backend.pipe_engine.schedule import *


# num_stages, num_micro_batches, max_new_tokens=5
def view_pipe_schedule(schedule_cls, stages, micro_batches, **kwargs):
    # max new tokens for
    scheds = [schedule_cls(micro_batches=micro_batches, stages=stages, stage_id=i) for i in range(stages)]

    print(f"Schedule: {schedule_cls.__name__}, stages: {stages}, micro_batches: {micro_batches}")
    for step_id, cmd_packs in enumerate(zip(*scheds)):
        print("=" * 30)
        print(f"Step {step_id}")
        for stage_id, cmd in enumerate(cmd_packs):
            print("-" * 30)
            print(f"Stage {stage_id}: {cmd}")


if __name__ == "__main__":
    view_pipe_schedule(TrainSchedule, stages=3, micro_batches=3)
    # view_pipe_schedule(InferenceSchedule, stages=3, micro_batches=3)
