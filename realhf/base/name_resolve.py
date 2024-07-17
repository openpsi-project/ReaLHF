# Implements a simple name resolving service, which can be considered as a distributed key-value dict.
import dataclasses
import getpass
import os
import queue
import random
import shutil
import socket
import threading
import time
import uuid
from typing import Callable, List, Optional

from realhf.base import logging, security, timeutil
from realhf.base.cluster import spec as cluster_spec

logger = logging.getLogger("name-resolve")


class ArgumentError(Exception):
    pass


class NameEntryExistsError(Exception):
    pass


class NameEntryNotFoundError(Exception):
    pass


class NameRecordRepository:

    def __del__(self):
        try:
            self.reset()
        except Exception as e:
            logger.info(f"Exception ignore when deleting NameResolveRepo {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset()

    def add(
        self,
        name,
        value,
        delete_on_exit=True,
        keepalive_ttl=None,
        replace=False,
    ):
        """Creates a name record in the central repository.

        In our semantics, the name record repository is essentially a multimap (i.e. Dict[str, Set[str]]).
        This class keeps a single name->value map, where the name can be non-unique while the value has to be.
        The class also deletes the (name, value) pair on exits (__exit__/__del__) if opted-in. In case of
        preventing unexpected exits (i.e. process crashes without calling graceful exits), an user may also
        want to specify time_to_live and call touch() regularly to allow a more consistent

        Args:
            name: The key of the record. It has to be a valid path-like string; e.g. "a/b/c". If the name
                already exists, the behaviour is defined by the `replace` argument.
            value: The value of the record. This can be any valid string.
            delete_on_exit: If the record shall be deleted when the repository closes.
            keepalive_ttl: If not None, adds a time-to-live in seconds for the record. The repository
                shall keep pinging the backend service with at least this frequency to make sure the name
                entry is alive during the lifetime of the repository. On the other hand, specifying this
                prevents stale keys caused by the scenario that a Python process accidentally crashes before
                calling delete().
            replace: If the name already exists, then replaces the current value with the supplied value if
                `replace` is True, or raises exception if `replace` is False.
        """
        raise NotImplementedError()

    def add_subentry(self, name, value, **kwargs):
        """Adds a sub-entry to the key-root `name`.

        The values is retrievable by get_subtree() given that no other
        entries use the name prefix.
        """
        sub_name = name.rstrip("/") + "/" + str(uuid.uuid4())[:8]
        self.add(sub_name, value, **kwargs)
        return sub_name

    def delete(self, name):
        """Deletes an existing record."""
        raise NotImplementedError()

    def clear_subtree(self, name_root):
        """Deletes all records whose names start with the path root name_root;
        specifically, whose name either is `name_root`, or starts with
        `name_root.rstrip("/") + "/"`."""
        raise NotImplementedError()

    def get(self, name):
        """Returns the value of the key.

        Raises NameEntryNotFoundError if not found.
        """
        raise NotImplementedError()

    def get_subtree(self, name_root):
        """Returns all values whose names start with the path root name_root;
        specifically, whose name either is `name_root`, or starts with
        `name_root.rstrip("/") + "/"`."""
        raise NotImplementedError()

    def find_subtree(self, name_root):
        """Returns all KEYS whose names start with the path root name_root."""
        raise NotImplementedError()

    def wait(self, name, timeout=None, poll_frequency=1):
        """Waits until a name appears.

        Raises:
             TimeoutError: if timeout exceeds.
        """
        start = time.monotonic()
        while True:
            try:
                return self.get(name)
            except NameEntryNotFoundError:
                pass
            if timeout is None or timeout > 0:
                time.sleep(
                    poll_frequency + random.random() * 0.1
                )  # To reduce concurrency.
            if timeout is not None and time.monotonic() - start > timeout:
                raise TimeoutError(
                    f"Timeout waiting for key '{name}' ({self.__class__.__name__})"
                )

    def reset(self):
        """Deletes all entries added via this repository instance's
        add(delete_on_exit=True)."""
        raise NotImplementedError()

    def watch_names(
        self,
        names: List,
        call_back: Callable,
        poll_frequency=15,
        wait_timeout=300,
    ):
        """Watch a name, execute call_back when key is deleted."""
        if isinstance(names, str):
            names = [names]

        q = queue.Queue(maxsize=len(names))
        for _ in range(len(names) - 1):
            q.put(0)

        def wrap_call_back():
            try:
                q.get_nowait()
            except queue.Empty:
                logger.info(f"Key {names} is gone. Executing callback {call_back}")
                call_back()

        for name in names:
            t = threading.Thread(
                target=self._watch_thread_run,
                args=(name, wrap_call_back, poll_frequency, wait_timeout),
                daemon=True,
            )
            t.start()

    def _watch_thread_run(self, name, call_back, poll_frequency, wait_timeout):
        self.wait(name, timeout=wait_timeout, poll_frequency=poll_frequency)
        while True:
            try:
                self.get(name)
                time.sleep(poll_frequency + random.random())
            except NameEntryNotFoundError:
                call_back()
                break


class MemoryNameRecordRepository(NameRecordRepository):
    """Stores all the records in a thread-local memory.

    Note that this is most likely for testing purposes:
    any distributed application is impossible to use this.
    """

    def __init__(self, log_events=False):
        self.__store = {}
        self.__log_events = log_events

    def add(
        self,
        name,
        value,
        delete_on_exit=True,
        keepalive_ttl=None,
        replace=False,
    ):
        if self.__log_events:
            print(f"NameResolve: add {name} {value}")
        if name in self.__store and not replace:
            raise NameEntryExistsError(f"K={name} V={self.__store[name]} V2={value}")
        assert isinstance(value, str)
        self.__store[name] = value

    def touch(self, name, value, new_time_to_live):
        raise NotImplementedError()

    def delete(self, name):
        if self.__log_events:
            print(f"NameResolve: delete {name}")
        if name not in self.__store:
            raise NameEntryNotFoundError(f"K={name}")
        del self.__store[name]

    def clear_subtree(self, name_root):
        if self.__log_events:
            print(f"NameResolve: clear_subtree {name_root}")
        name_root = name_root.rstrip("/")
        for name in list(self.__store):
            if (
                name_root == "/"
                or name == name_root
                or name.startswith(name_root + "/")
            ):
                del self.__store[name]

    def get(self, name):
        if name not in self.__store:
            raise NameEntryNotFoundError(f"K={name}")
        r = self.__store[name]
        if self.__log_events:
            print(f"NameResolve: get {name} -> {r}")
        return r

    def get_subtree(self, name_root):
        if self.__log_events:
            print(f"NameResolve: get_subtree {name_root}")
        name_root = name_root.rstrip("/")
        rs = []
        for name, value in self.__store.items():
            if (
                name_root == "/"
                or name == name_root
                or name.startswith(name_root + "/")
            ):
                rs.append(value)
        return rs

    def find_subtree(self, name_root):
        if self.__log_events:
            print(f"NameResolve: find_subtree {name_root}")
        rs = []
        for name, value in self.__store.items():
            if name.startswith(name_root):
                rs.append(name)
        rs.sort()
        return rs

    def reset(self):
        self.__store = {}


class NfsNameRecordRepository(NameRecordRepository):
    RECORD_ROOT = f"{cluster_spec.fileroot}/name_resolve/"

    def __init__(self, **kwargs):
        self.__to_delete = set()

    @staticmethod
    def __dir_path(name):
        return os.path.join(NfsNameRecordRepository.RECORD_ROOT, name)

    @staticmethod
    def __file_path(name):
        return os.path.join(NfsNameRecordRepository.__dir_path(name), "ENTRY")

    def add(
        self,
        name,
        value,
        delete_on_exit=True,
        keepalive_ttl=None,
        replace=False,
    ):
        path = self.__file_path(name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.isfile(path) and not replace:
            raise NameEntryExistsError(path)
        local_id = str(uuid.uuid4())[:8]
        with open(path + f".tmp.{local_id}", "w") as f:
            f.write(str(value))
        os.rename(path + f".tmp.{local_id}", path)
        if delete_on_exit:
            self.__to_delete.add(name)

    def delete(self, name):
        path = self.__file_path(name)
        if not os.path.isfile(path):
            raise NameEntryNotFoundError(path)
        os.remove(path)
        while True:
            path = os.path.dirname(path)
            if path == NfsNameRecordRepository.RECORD_ROOT:
                break
            if len(os.listdir(path)) > 0:
                break
            shutil.rmtree(path, ignore_errors=True)
        if name in self.__to_delete:
            self.__to_delete.remove(name)

    def clear_subtree(self, name_root):
        dir_path = self.__dir_path(name_root)
        if os.path.isdir(dir_path):
            logger.info("Removing name resolve path: %s", dir_path)
            shutil.rmtree(dir_path)
        else:
            logger.info("No such name resolve path: %s", dir_path)

    def get(self, name):
        path = self.__file_path(name)
        if not os.path.isfile(path):
            raise NameEntryNotFoundError(path)
        with open(path, "r") as f:
            return f.read().strip()

    def get_subtree(self, name_root):
        dir_path = self.__dir_path(name_root)
        rs = []
        if os.path.isdir(dir_path):
            for item in os.listdir(dir_path):
                try:
                    rs.append(self.get(os.path.join(name_root, item)))
                except NameEntryNotFoundError:
                    pass
        return rs

    def find_subtree(self, name_root):
        dir_path = self.__dir_path(name_root)
        rs = []
        if os.path.isdir(dir_path):
            for item in os.listdir(dir_path):
                rs.append(os.path.join(name_root, item))
        rs.sort()
        return rs

    def reset(self):
        for name in list(self.__to_delete):
            try:
                self.delete(name)
            except:
                pass
        self.__to_delete = {}


class RedisNameRecordRepository(NameRecordRepository):
    _IS_FRL = socket.gethostname().startswith("frl")
    REDIS_HOST = "redis" if _IS_FRL else "localhost"
    REDIS_PASSWORD = security.read_key("redis") if _IS_FRL else None
    REDIS_DB = 0
    KEEPALIVE_POLL_FREQUENCY = 1

    @dataclasses.dataclass
    class _Entry:
        value: str
        keepalive_ttl: Optional[int] = None
        keeper: Optional[timeutil.FrequencyControl] = None

    def __init__(self, **kwargs):
        import redis
        from redis.backoff import ExponentialBackoff
        from redis.retry import Retry

        super().__init__()
        self.__lock = threading.Lock()
        self.__redis = redis.Redis(
            host=RedisNameRecordRepository.REDIS_HOST,
            password=RedisNameRecordRepository.REDIS_PASSWORD,
            db=RedisNameRecordRepository.REDIS_DB,
            socket_timeout=60,
            retry_on_timeout=True,
            retry=Retry(ExponentialBackoff(180, 60), 3),
        )
        self.__entries = {}
        self.__keepalive_running = True
        self.__keepalive_thread = threading.Thread(
            target=self.__keepalive_thread_run, daemon=True
        )
        self.__keepalive_thread.start()

    def __del__(self):
        self.__keepalive_running = False
        self.__keepalive_thread.join(timeout=5)
        self.reset()
        self.__redis.close()

    def add(self, name, value, delete_on_exit=True, keepalive_ttl=10, replace=False):
        # deprecated parameter: delete_on_exit, now every entry has a default keepalive_ttl=10 seconds
        if name.endswith("/"):
            raise ValueError(f"Entry name cannot end with '/': {name}")

        with self.__lock:
            keepalive_ttl = int(keepalive_ttl * 1000)
            assert (
                keepalive_ttl > 0
            ), f"keepalive_ttl in milliseconds must >0: {keepalive_ttl}"
            if self.__redis.set(name, value, px=keepalive_ttl, nx=not replace) is None:
                raise NameEntryExistsError(f"Cannot set Redis key: K={name} V={value}")

            # touch every 1/3 of keepalive_ttl to prevent Redis from deleting the key
            # after program exit, redis will automatically delete key in keepalive_ttl
            self.__entries[name] = self._Entry(
                value=value,
                keepalive_ttl=keepalive_ttl,
                keeper=timeutil.FrequencyControl(
                    frequency_seconds=keepalive_ttl / 1000 / 3
                ),
            )

    def delete(self, name):
        with self.__lock:
            self.__delete_locked(name)

    def __delete_locked(self, name):
        if name in self.__entries:
            del self.__entries[name]
        if self.__redis.delete(name) == 0:
            raise NameEntryNotFoundError(f"No such Redis entry to delete: {name}")

    def clear_subtree(self, name_root):
        with self.__lock:
            count = 0
            for name in list(self.__find_subtree_locked(name_root)):
                try:
                    self.__delete_locked(name)
                    count += 1
                except NameEntryNotFoundError:
                    pass
            logger.info("Deleted %d Redis entries under %s", count, name_root)

    def get(self, name):
        with self.__lock:
            return self.__get_locked(name)

    def __get_locked(self, name):
        r = self.__redis.get(name)
        if r is None:
            raise NameEntryNotFoundError(f"No such Redis entry: {name}")
        return r.decode()

    def get_subtree(self, name_root):
        with self.__lock:
            rs = []
            for name in self.__find_subtree_locked(name_root):
                rs.append(self.__get_locked(name))
            rs.sort()
            return rs

    def find_subtree(self, name_root):
        with self.__lock:
            return list(sorted(self.__find_subtree_locked(name_root)))

    def reset(self):
        with self.__lock:
            count = 0
            for name in list(self.__entries):
                try:
                    self.__delete_locked(name)
                    count += 1
                except NameEntryNotFoundError:
                    pass
            self.__entries = {}
            logger.info("Reset %d saved Redis entries", count)

    def __keepalive_thread_run(self):
        while self.__keepalive_running:
            time.sleep(self.KEEPALIVE_POLL_FREQUENCY)
            with self.__lock:
                for name, entry in self.__entries.items():
                    if entry.keeper is not None and entry.keeper.check():
                        r = self.__redis.set(name, entry.value, px=entry.keepalive_ttl)
                        if r is None:
                            logger.error(
                                "Failed touching Redis key: K=%s V=%s",
                                name,
                                entry.value,
                            )

    def __find_subtree_locked(self, name_root):
        pattern = name_root + "*"
        return [k.decode() for k in self.__redis.keys(pattern=pattern)]

    def _testonly_drop_cached_entry(self, name):
        """Used by unittest only to simulate the case that the Python process
        crashes and the key is automatically removed after TTL."""
        with self.__lock:
            del self.__entries[name]
            print("Testonly: dropped key:", name)


def make_repository(type_="nfs", **kwargs):
    if type_ == "memory":
        return MemoryNameRecordRepository(**kwargs)
    elif type_ == "nfs":
        return NfsNameRecordRepository(**kwargs)
    elif type_ == "redis":
        return RedisNameRecordRepository(**kwargs)
    else:
        raise NotImplementedError(f"No such name resolver: {type_}")


# DEFAULT_REPOSITORY_TYPE = "redis" if socket.gethostname().startswith("frl") else "nfs"
DEFAULT_REPOSITORY_TYPE = "nfs"
DEFAULT_REPOSITORY = make_repository(DEFAULT_REPOSITORY_TYPE)
add = DEFAULT_REPOSITORY.add
add_subentry = DEFAULT_REPOSITORY.add_subentry
delete = DEFAULT_REPOSITORY.delete
clear_subtree = DEFAULT_REPOSITORY.clear_subtree
get = DEFAULT_REPOSITORY.get
get_subtree = DEFAULT_REPOSITORY.get_subtree
find_subtree = DEFAULT_REPOSITORY.find_subtree
wait = DEFAULT_REPOSITORY.wait
reset = DEFAULT_REPOSITORY.reset
watch_names = DEFAULT_REPOSITORY.watch_names


def reconfigure(*args, **kwargs):
    global DEFAULT_REPOSITORY, DEFAULT_REPOSITORY_TYPE
    global add, add_subentry, delete, clear_subtree, get, get_subtree, find_subtree, wait, reset, watch_names
    DEFAULT_REPOSITORY = make_repository(*args, **kwargs)
    DEFAULT_REPOSITORY_TYPE = args[0]
    add = DEFAULT_REPOSITORY.add
    add_subentry = DEFAULT_REPOSITORY.add_subentry
    delete = DEFAULT_REPOSITORY.delete
    clear_subtree = DEFAULT_REPOSITORY.clear_subtree
    get = DEFAULT_REPOSITORY.get
    get_subtree = DEFAULT_REPOSITORY.get_subtree
    find_subtree = DEFAULT_REPOSITORY.find_subtree
    wait = DEFAULT_REPOSITORY.wait
    reset = DEFAULT_REPOSITORY.reset
    watch_names = DEFAULT_REPOSITORY.watch_names
