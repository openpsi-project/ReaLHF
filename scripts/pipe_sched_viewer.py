import sys

sys.path.append("../")

from impl.model.backend.ds_pipe_engine.schedule import *


def _is_even(x):
    return x % 2 == 0


def _is_odd(x):
    return x % 2 != 0


def _even_step_forward_id(step_id, stages, stage_id):
    base = step_id // 2
    micro_batch_id = int(base - stage_id // 2)
    return micro_batch_id


def _odd_step_forward_id(step_id, stages, stage_id):
    base = (step_id - 1) // 2
    micro_batch_id = int(base - stage_id // 2)
    return micro_batch_id


def _even_step_backward_id(step_id, stages, stage_id):
    base = step_id // 2
    micro_batch_id = int(base - stages + (stage_id + 1) // 2)
    return micro_batch_id


def _odd_step_backward_id(step_id, stages, stage_id):
    base = ((step_id - 1) // 2) - stages + 1
    micro_batch_id = int(base + stage_id // 2)
    return micro_batch_id


def _step_to_micro_batch(step_id, stages, stage_id):
    if _is_even(step_id) and _is_even(stage_id):
        micro_batch_id = _even_step_forward_id(step_id, stages, stage_id)
        is_forward = True

    elif _is_odd(step_id) and _is_odd(stage_id):
        micro_batch_id = _odd_step_forward_id(step_id, stages, stage_id)
        is_forward = True

    elif _is_even(step_id) and _is_odd(stage_id):
        micro_batch_id = _even_step_backward_id(step_id, stages, stage_id)
        is_forward = False

    elif _is_odd(step_id) and _is_even(stage_id):
        micro_batch_id = _odd_step_backward_id(step_id, stages, stage_id)
        is_forward = False

    else:
        assert False

    return micro_batch_id, is_forward


# num_stages, num_micro_batches, max_new_tokens=5
def view_pipe_schedule(schedule_cls, stages, micro_batches, **kwargs):
    # max new tokens for
    scheds = [
        schedule_cls(micro_batches=micro_batches, stages=stages, stage_id=i, **kwargs) for i in range(stages)
    ]

    print(f"Schedule: {schedule_cls.__name__}, stages: {stages}, micro_batches: {micro_batches}")
    for step_id, cmd_packs in enumerate(zip(*scheds)):
        print("=" * 30)
        print(f"Step {step_id}")
        for stage_id, cmd in enumerate(cmd_packs):
            print("-" * 30)
            mbid = _step_to_micro_batch(step_id, stages, stage_id)[0]
            print(f"Stage {stage_id}, micro batch {mbid}: {cmd}")


if __name__ == "__main__":
    view_pipe_schedule(GenerateSchedule, stages=3, micro_batches=3, max_new_tokens=3)
    # view_pipe_schedule(InferenceSchedule, stages=3, micro_batches=3)
