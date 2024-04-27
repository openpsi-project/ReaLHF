import itertools
import os

from reallm.base.monitor import get_tracer
from reallm.impl.model.backend.pipe_engine.instruction import BackwardPass, ForwardPass, InstructionSet

if __name__ == "__main__":
    os.environ["REAL_TRACE"] = "1"
    tracer = get_tracer(
        tracer_entries=int(2e6),
        # max_stack_depth=10,
        ignore_c_function=False,
        ignore_frozen=True,
        log_async=True,
        output_file=f"/home/meizy/logs/viztracer/others/inst_set.json")
    tracer.start()
    insts = list(itertools.chain([[ForwardPass(i, i + 1), BackwardPass(i + 2, i + 3)] for i in range(100)]))
    inst_set = InstructionSet()
    inst_set.add(insts)

    for _ in range(10):
        inst_set.find()

    tracer.save()
