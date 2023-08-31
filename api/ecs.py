from typing import Any, Callable, Dict, get_type_hints, Iterable, List, Optional, Tuple, Type, Union
from unittest.mock import MagicMock
import collections
import concurrent.futures
import dataclasses
import functools
import inspect
import itertools
import logging
import threading

import torch

import base.namedarray as namedarray

logger = logging.getLogger("MasterWorker ECS")


class ModelQueryMeta(type):
    """Meta-class to make type hints like ModelQuery["my_model_name"] work."""

    def __getitem__(self, name: str):
        return functools.partial(ModelQuery, name)


class ModelQuery(metaclass=ModelQueryMeta):
    """Model query corresponding to the model_name in model worker configs."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    # The following methods are stubs for ECS validation.
    # These methods of a real model will have the signature:
    # fn(x: namedarray.NamedArray) -> namedarray.NamedArray
    def __call__(self, data: namedarray.NamedArray):  # aka inference
        return collections.defaultdict(lambda: torch.zeros(1))

    def train_step(self, data: namedarray.NamedArray):
        return collections.defaultdict(lambda: torch.zeros(1))

    def generate(self, data: namedarray.NamedArray):
        return collections.defaultdict(lambda: torch.zeros(1))

    def evaluate(self, data: namedarray.NamedArray):
        return collections.defaultdict(lambda: torch.zeros(1))


# TODO: add shape in data query
# TODO: multiple string query returns dict
class DataQueryMeta(type):
    """Meta-class to make type hints like DataQuery["my_data_name"] work."""

    def __getitem__(self, name: str):
        return functools.partial(DataQuery, name)


class DataQuery(metaclass=DataQueryMeta):
    """Data query for data created by models."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    def __setattr__(self, loc, value):
        if loc != '_name':
            raise RuntimeError("DataQuery should not be modified in-place."
                               " Use set_data in Commands instead.")
        super().__setattr__(loc, value)


class RawDataQueryMeta(type):
    """Meta-class to make type hints like RawDataQuery["my_data_name"] work.
    
    Raw data is data loaded from dataset.
    """

    def __getitem__(self, name: str):
        return functools.partial(RawDataQuery, name)


class RawDataQuery(metaclass=RawDataQueryMeta):
    """Data query for raw data loaded from dataset."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    def __setattr__(self, loc, value):
        if loc != '_name':
            raise RuntimeError("RawDataQuery should not be modified in-place."
                               " Use set_data in Commands instead.")
        super().__setattr__(loc, value)


class Commands:
    """Method stubs of master worker."""

    def __init__(self):
        self.lock = threading.Lock()
        self.data_registry = {}

        self._epoch = self._epoch_step = self._global_step = 0

    def get_data(self, key: str) -> Any:
        with self.lock:
            return self.data_registry[key]

    def set_data(self, key: str, value: Any):
        # We must explicitly use a key here, otherwise we cannot validate data dependency.
        with self.lock:
            if key in self.data_registry:
                raise RuntimeError("Data name conflict.")
            self.data_registry[key] = value

    def build_model_inputs(self, metadata: Optional[Dict] = None, **kwargs) -> namedarray.NamedArray:
        x = namedarray.from_dict(kwargs)
        x.register_metadata(epoch=self._epoch, epoch_step=self._epoch_step, global_step=self._global_step)
        if metadata is not None:
            x.register_metadata(**metadata)
        return x

    def _update_counter(self, epoch, epoch_step, global_step):
        self._epoch = epoch
        self._epoch_step = epoch_step
        self._global_step = global_step

    def log(self, *results):
        # TODO: wandb log
        for res in results:
            logger.info(f"Train result: {res}")


@dataclasses.dataclass
class MasterWorkerExecutable:
    levels: List[List[str]]
    funcs: List[List[Callable]]


class MasterWorkerECS:

    def __init__(self, model_worker_configs):
        self._model_names = list(set([mw.model_name for mw in model_worker_configs]))
        self._system_registry: Dict[str, Tuple] = collections.OrderedDict()

    def add_system(
        self,
        func: Callable,
        before: Optional[Iterable[Callable]] = None,
        after: Optional[Iterable[Callable]] = None,
    ):
        self._validate_system_func(func)

        if before is None:
            before = []
        elif not isinstance(before, Iterable):
            before = [before]
        if after is None:
            after = []
        elif not isinstance(after, Iterable):
            after = [after]

        if func.__name__ not in self._system_registry:
            # key=task_name, value=(task_func, [task_names that depends on this task])
            self._system_registry[func.__name__] = (func, list(set([f.__name__ for f in before])))
        else:
            # This means that the task's dependencies have created a entry
            _, before_deps = self._system_registry[func.__name__]
            self._system_registry[func.__name__] = (func, list(set(before_deps + [f.__name__
                                                                                  for f in before])))

        for f in after:
            if f.__name__ not in self._system_registry:
                self._system_registry[f.__name__] = (None, [func.__name__])
            else:
                after_func, after_deps = self._system_registry[f.__name__]
                after_deps = list(set(after_deps + [func.__name__]))
                self._system_registry[f.__name__] = (after_func, after_deps)
        return self

    def add_systems(self,
                    funcs: List[Callable],
                    before: Optional[Iterable[Callable]] = None,
                    after: Optional[Iterable[Callable]] = None):
        for func in funcs:
            self.add_system(func, before, after)
        return self

    def resolve_order(self) -> List[List[str]]:
        for k, v in self._system_registry.items():
            if not v[0]:
                raise ValueError(f"There exist systems depending on {k} but the callable is not registered.")

        self._resolve_implied_dependencies()

        in_degrees = collections.defaultdict(int)
        for task in self._system_registry:
            in_degrees[task] += 0
            for dependency in self._system_registry[task][1]:
                in_degrees[dependency] += 1
        in_degrees = dict(in_degrees)

        levels = []
        current_level = []
        next_level = []

        for task_name, in_degree in in_degrees.items():
            if in_degree == 0:
                current_level.append(task_name)

        while current_level:
            levels.append(current_level)
            for task in current_level:
                for dependency in self._system_registry[task][1]:
                    in_degrees[dependency] -= 1
                    if in_degrees[dependency] == 0:
                        next_level.append(dependency)

            current_level = next_level
            next_level = []

        if len(set(itertools.chain.from_iterable(levels))) != len(self._system_registry):
            raise ValueError("There is a circular dependency in the system registry.")

        logger.info(f"Task levels resolved by ECS: {levels}")
        return levels

    def _resolve_implied_dependencies(self):
        raw_data_entries = []
        # Data required have two sources: DataQuery parameters and Commands.get_data() calls
        required_data_entries = collections.defaultdict(list)
        # Data generated have one source: Commands.set_data() calls
        generated_data_entries = collections.defaultdict(list)

        for task, (func, *_) in self._system_registry.items():
            type_hints = get_type_hints(func)
            for type_hint in type_hints.values():
                if isinstance(type_hint(), DataQuery):
                    required_data_entries[task].append(type_hint().name)
                elif isinstance(type_hint(), RawDataQuery):
                    raw_data_entries.append(type_hint().name)
            # Trace calls on Commands (i.e., get_data and set_data) using mocks.
            command_argument_indices = [
                i for i, t in enumerate(type_hints.values()) if isinstance(t(), Commands)
            ]
            arguments = [MagicMock() for _ in type_hints]
            func(*arguments)
            for idx in command_argument_indices:
                m = arguments[idx]
                required_data_entries[task] += [x.args[0] for x in m.get_data.call_args_list]
                generated_data_entries[task] += [x.args[0] for x in m.set_data.call_args_list]

        generated_data_entries = {k: list(set(v)) for k, v in generated_data_entries.items()}
        required_data_entries = {k: list(set(v)) for k, v in required_data_entries.items()}

        for task, (_, deps) in self._system_registry.items():
            generated_date_keys = generated_data_entries[task]
            for k in generated_date_keys:
                for child_task in self._system_registry:
                    if child_task == task:
                        continue
                    if k in required_data_entries[child_task] and child_task not in deps:
                        deps.append(child_task)
                        logger.info(f"Dependency added: {task} -> {child_task} because of data entry `{k}`.")
        for task, (_, deps) in self._system_registry.items():
            logger.info(f"Dependency: {task} -> {deps}.")

    def build(self):
        levels = self.resolve_order()
        try:
            # TODO: if all data queries provide shapes, replace DataQuery with shaped torch.Tensor instead of Mock()
            # TODO: do not log in mock run
            self._mock_run(levels)
        except Exception as e:
            raise RuntimeError("Mock run failed because of the reason above.") from e
        return MasterWorkerExecutable(levels, [[self._system_registry[task][0] for task in level_tasks]
                                               for level_tasks in levels])

    def _validate_system_func(self, func):
        params = inspect.signature(func).parameters
        type_hints = get_type_hints(func)
        if len(type_hints) != len(params):
            raise RuntimeError(
                "Registered system function misses at least one type hints. "
                "All arguments should have type hints of ModelQuery, DataQuery, RawDataQuery, or Commands.")
        for k, type_hint in type_hints.items():
            t = type_hint()
            if not isinstance(t, (ModelQuery, DataQuery, RawDataQuery, Commands)):
                raise RuntimeError(
                    f"Error when parsing argument {k} of function {func}. "
                    f"Only ModelQuery, DataQuery, RawDataQuery and Commands are allowed as system functions.")
            if isinstance(t, ModelQuery) and t.name not in self._model_names:
                raise RuntimeError(f"Model query named `{t.name}` in function `{func.__name__}` "
                                   "not found in model worker configs.")

    def _mock_run(self, levels):

        def _build_duck_arguments(func, duck_model, duck_data, duck_raw_data, duck_commands):
            # Only used for validation. In real execution, if multiple model queries are used,
            # the duck models are different from each other. We can't build arguments in this way.
            type_hints = get_type_hints(func)
            arguments = []
            for type_hint in type_hints.values():
                if isinstance(type_hint(), ModelQuery):
                    arguments.append(duck_model)
                elif isinstance(type_hint(), DataQuery):
                    arguments.append(duck_data)
                elif isinstance(type_hint(), RawDataQuery):
                    arguments.append(duck_raw_data)
                elif isinstance(type_hint(), Commands):
                    arguments.append(duck_commands)
            return arguments

        def _assert_data_query_exists(global_data_registry, func):
            type_hints = get_type_hints(func)
            for type_hint in type_hints.values():
                t = type_hint()
                if isinstance(t, DataQuery) and t.name not in global_data_registry:
                    raise RuntimeError(
                        f"Data query named `{t.name}` in function `{func.__name__}` has not been registered. "
                        "Please check whether the data entry created by `commands.set_data` has the same name as the given one. "
                        "For data loaded from dataset rather than created by models, use RawDataQuery instead. "
                        f"Registered data queries: {list(global_data_registry.keys())}.")
                elif isinstance(t, RawDataQuery) and t.name not in global_data_registry:
                    global_data_registry[t.name] = None

        duck_commands = Commands()
        duck_model = MagicMock()

        func_levels = [[self._system_registry[task][0] for task in level_tasks] for level_tasks in levels]
        max_concurrency = max(len(tasks) for tasks in levels)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            for funcs in func_levels:
                futures = []
                for func in funcs:
                    _assert_data_query_exists(duck_commands.data_registry, func)
                    arguments = _build_duck_arguments(func, duck_model, MagicMock(), MagicMock(),
                                                      duck_commands)
                    futures.append(executor.submit(func, *arguments))
                [future.result() for future in futures]
                method_names = [x[0] for x in duck_model.mock_calls]
                for method_name in method_names:
                    if all((x not in method_name) for x in ['generate', 'evaluate', 'train_step', '']):
                        raise RuntimeError(f"Unknown model API call: {method_name} when "
                                           f"executing {[func.__name__ for func in funcs]}.")
