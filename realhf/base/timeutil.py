import dataclasses
import math
import threading
import time
from abc import ABC
from typing import Callable, List

INFINITE_DURATION = 60 * 60 * 24 * 365 * 1000


class FrequencyControl:
    """An utility to control the execution of code with a time or/and step
    frequency."""

    def __init__(
        self, frequency_seconds=None, frequency_steps=None, initial_value=False
    ):
        """Initialization method of FrequencyControl.
        Args:
            frequency_seconds: Minimal interval between two trigger.
            frequency_steps: Minimal number of steps between two triggers.
            initial_value: In true, the first call of check() returns True.

        NOTE:
            - If both frequency_seconds and frequency_steps are None, the checking will always return False except
             for the specified initial value.
            - If passed both, both frequency and steps conditions have to be met for check() to return True.
            - If one is passed, checking on the other condition will be ignored.
        """
        self.frequency_seconds = frequency_seconds
        self.frequency_steps = frequency_steps
        self.__start_time = time.monotonic()
        self.__steps = 0
        self.__last_time = time.monotonic()
        self.__last_steps = 0
        self.__interval_seconds = self.__interval_steps = None
        self.__initial_value = initial_value
        self.__lock = threading.Lock()

    @property
    def total_seconds(self):
        return time.monotonic() - self.__start_time

    @property
    def total_steps(self):
        return self.__steps

    @property
    def interval_seconds(self):
        return self.__interval_seconds

    @property
    def interval_steps(self):
        return self.__interval_steps

    def check(self, steps=1):
        """Check whether frequency condition is met.
        Args:
            steps: number of step between this and the last call of check()

        Returns:
            flag: True if condition is met, False other wise
        """
        with self.__lock:
            now = time.monotonic()
            self.__steps += steps

            if self.__initial_value:
                self.__last_time = now
                self.__last_steps = self.__steps
                self.__initial_value = False
                return True

            self.__interval_seconds = now - self.__last_time
            self.__interval_steps = self.__steps - self.__last_steps
            if self.frequency_steps is None and self.frequency_seconds is None:
                return False
            if (
                self.frequency_seconds is not None
                and self.__interval_seconds < self.frequency_seconds
            ):
                return False
            if (
                self.frequency_steps is not None
                and self.__interval_steps < self.frequency_steps
            ):
                return False
            self.__last_time = now
            self.__last_steps = self.__steps

            return True

    def reset_time(self):
        self.__last_time = time.monotonic()


@dataclasses.dataclass
class EpochStepTimeFreqCtl:
    freq_epoch: int
    freq_step: int
    freq_sec: int

    def __post_init__(self):
        self.epoch_ctl = FrequencyControl(frequency_steps=self.freq_epoch)
        self.step_ctl = FrequencyControl(frequency_steps=self.freq_step)
        self.time_ctl = FrequencyControl(frequency_seconds=self.freq_sec)

    def check(self, epochs: int, steps: int):
        x, y, z = (
            self.epoch_ctl.check(epochs),
            self.step_ctl.check(steps),
            self.time_ctl.check(),
        )
        return x or y or z


@dataclasses.dataclass
class Scheduler(ABC):
    init_value: float
    total_iters: int

    def __post_init__(self):
        if self.total_iters <= 0:
            raise ValueError("total_iters should be a positive number.")

    def get(self, step: int) -> float:
        """Get the scheduled value at the current `step`."""
        if step < 0 or step > self.total_iters:
            raise ValueError(
                f"Scheduler step should be in the interval [0, {self.total_iters}]. Input {step}."
            )
        return self._get(step)

    def _get(self, step):
        raise NotImplementedError()

    @property
    def final_value(self):
        return self.get(step=self.total_iters)


@dataclasses.dataclass
class ConstantScheduler(Scheduler):

    def _get(self, *args, **kwargs) -> float:
        return self.init_value


@dataclasses.dataclass
class LinearScheduler(Scheduler):
    end_value: float

    def _get(self, step: int) -> float:
        return (
            self.end_value - self.init_value
        ) / self.total_iters * step + self.init_value


@dataclasses.dataclass
class ExponentialScheduler(Scheduler):
    decay: float

    def _get(self, step: int) -> float:
        return self.init_value * self.decay**step


@dataclasses.dataclass
class CosineDecayScheduler(Scheduler):
    end_value: float

    def __post_init__(self):
        super().__post_init__()
        if self.end_value >= self.init_value:
            raise ValueError("end_value should be smaller than init_value!")

    def _get(self, step: int) -> float:
        delta = self.init_value - self.end_value
        return (
            delta * 0.5 * (1 + math.cos(math.pi / self.total_iters * step))
            + self.end_value
        )


@dataclasses.dataclass
class ChainedScheduler:
    schedulers: List[Scheduler]

    @property
    def total_iters(self):
        return sum(x.total_iters for x in self.schedulers)

    @property
    def init_value(self):
        return self.schedulers[0].init_value

    @property
    def final_value(self):
        return self.schedulers[-1].final_value

    def __post_init__(self):
        for i in range(len(self.schedulers) - 1):
            # Float point err 1e-8.
            if (
                abs(self.schedulers[i + 1].get(0) - self.schedulers[i].final_value)
                > 1e-8
            ):
                raise ValueError(
                    f"Values should be consecutive between "
                    f"the {i}-th ({type(self.schedulers[i])}) and "
                    f"the {i+1}-th {type(self.schedulers[i+1])} schedulers! "
                    f"End value is {self.schedulers[i].final_value} and the "
                    f"next init value is {self.schedulers[i + 1].get(0)}."
                )

    def get(self, step: int) -> float:
        for s in self.schedulers:
            if step > s.total_iters:
                step -= s.total_iters
            else:
                return s.get(step)
