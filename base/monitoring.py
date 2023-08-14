from typing import Dict, List
import dataclasses
import logging
import os
import platform
import subprocess
import threading
import time
import wsgiref.simple_server

import prometheus_client
import psutil
import wandb

import base.cluster

logger = logging.getLogger("monitoring")


class _SilentHandler(wsgiref.simple_server.WSGIRequestHandler):
    """WSGI handler that does not log requests.
    """

    def log_message(self, *args):
        pass


def start_prometheus_server():
    """Returns listen port.
    """
    app = prometheus_client.make_wsgi_app()
    httpd = wsgiref.simple_server.make_server('', 0, app, handler_class=_SilentHandler)
    port = httpd.socket.getsockname()[1]
    t = threading.Thread(target=httpd.serve_forever)
    t.daemon = True
    t.start()
    return port


class _TargetGroupHolder:

    def __init__(self, repo, group_name, delete_on_exit=True):
        self.repo = repo
        self.group_name = group_name
        self.__to_delete = delete_on_exit

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.delete()

    def __del__(self):
        self.delete()

    def delete(self):
        if self.__to_delete:
            self.repo.delete_target_group(self.group_name)
            self.__to_delete = False


class TargetRepository:
    TARGET_FILE_DIR = "/data/aigc/llm/monitoring/targets"
    # TARGET_FILE_DIR = "/data/aigc/llm/monitoring/targets"
    os.makedirs(TARGET_FILE_DIR, exist_ok=True)

    def __init__(self):
        self.__enabled = os.path.isdir(TargetRepository.TARGET_FILE_DIR)

    def add_target_group(self, group_name, target_addresses, delete_on_exit=True) -> _TargetGroupHolder:
        if not self.__enabled:
            return _TargetGroupHolder(None, None, delete_on_exit=False)
        group_name = "marl_exp." + os.environ.get("SLURM_JOBID", group_name)
        path = os.path.join(TargetRepository.TARGET_FILE_DIR, f"{group_name}.yaml")
        if os.path.isfile(path):
            try:
                os.remove(path)
            except PermissionError:
                logger.error("Failed to remove previous monitoring target group. This is likely due to "
                             "duplicated trial_name between different users.")
                raise FileExistsError(path)
        with open(path + ".tmp", "w") as f:
            f.write("- targets:\n")
            for address in target_addresses:
                f.write(f"  - '{address}'\n")
        os.rename(path + ".tmp", path)
        logger.info("Add monitoring group for %d targets: %s", len(target_addresses), path)
        return _TargetGroupHolder(self, group_name, delete_on_exit=delete_on_exit)

    def delete_target_group(self, group_name):
        if not self.__enabled:
            return
        path = os.path.join(TargetRepository.TARGET_FILE_DIR, f"{group_name}.yaml")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        logger.info("Delete monitoring group: %s", path)
        os.remove(path)


class PrometheusMetricViewer:

    def __init__(self, metric):
        assert isinstance(metric, prometheus_client.Summary) or isinstance(metric, prometheus_client.Counter)
        if metric._is_parent():
            values = list(metric._metrics.values())
            assert len(values) == 1, values
            metric = values[0]
        self.__metric: prometheus_client.Summary = metric
        self.__last_time_ns = time.monotonic_ns()
        self.__last_sum = self.__last_count = 0

    def rate(self):
        now = time.monotonic_ns()
        duration = (now - self.__last_time_ns) / 1e9
        self.__last_time_ns = now
        if isinstance(self.__metric, prometheus_client.Counter):
            s = self.__metric._value.get()
            r = (s - self.__last_sum) / duration
            self.__last_sum = s
        elif isinstance(self.__metric, prometheus_client.Summary):
            s = self.__metric._sum.get()
            n = self.__metric._count.get()
            if n == self.__last_count:
                r = 0.0
            else:
                r = (s - self.__last_sum) / (n - self.__last_count)
            self.__last_sum = s
            self.__last_count = n
        else:
            raise NotImplementedError()
        return r


DEFAULT_PROMETHEUS_LABEL_NAMES = ["host", "experiment", "trial", "worker", "worker_id", "policy"]


@dataclasses.dataclass
class MonitorInfo:
    """ Monitoring information for one worker, used to intialize monitor metrics and configs for an Monitor instance.
    
    Args:
        prometheus_labels: dict containing labels for prometheus, keys should be identical to prometheus_label_names
        prometheus_metrics: dict containing all metrics should be initialized for prometheus in Monitor class.
                            format: dict(metric_name=metric_type, ...),
                            metric_type could be: Histogram, Summary, Counter, Gauge.
        wandb_args: dict containing labels for wandb logging, same contents as self.__wandb_args in worker_base.py
        prometheus_label_names: list of names of labels in all metrics for a worker.
        std_output: whether to print monitor contents into logging file.
    """
    prometheus_labels: Dict
    if_log_wandb: bool
    wandb_args: Dict
    prometheus_metrics: Dict = dataclasses.field(default_factory=dict)
    prometheus_label_names: List[str] = dataclasses.field(
        default_factory=lambda: DEFAULT_PROMETHEUS_LABEL_NAMES)
    file_output: bool = False


class Monitor:
    """ Monitoring utility class.
    """

    def __init__(self, metrics: MonitorInfo):
        self.__wandb_run = None  # Lazy initialized when self.log_wandb is called.
        self.__wandb_args = metrics.wandb_args
        self.__if_log_wandb = metrics.if_log_wandb
        self.__prometheus_labels = metrics.prometheus_labels
        self.__prometheus_label_names = metrics.prometheus_label_names
        self.__prometheus_metrics = dict()
        self.__labeled_metrics = dict()

        for metric_name, metric_type in metrics.prometheus_metrics.items():
            self.__add_prometheus_metric(metric_name, metric_type)

    def __add_prometheus_metric(self, metric_name, metric_type):
        cls_ = None
        if metric_type == "Summary":
            cls_ = prometheus_client.Summary
        elif metric_type == "Counter":
            cls_ = prometheus_client.Counter
        elif metric_type == "Gauge":
            cls_ = prometheus_client.Gauge
        elif metric_type == "Histogram":
            cls_ = prometheus_client.Histogram
        else:
            raise Exception("Unknown prometheus metric type.")

        metric = cls_(metric_name, "", self.__prometheus_label_names)
        labeled = metric.labels(**self.__prometheus_labels)
        self.__prometheus_metrics[metric_name] = metric
        self.__labeled_metrics[metric_name] = labeled

    def update_metrics(self, metric_specs: Dict):
        for metric_name, metric_type in metric_specs.items():
            self.__add_prometheus_metric(metric_name, metric_type)

    def metric(self, metric_name: str):
        return self.__labeled_metrics[metric_name]

    def new_wandb_run(self, new_wandb_args):
        self.__wandb_run.finish()
        self.__wandb_run = None
        self.__wandb_args.update(new_wandb_args)

    def log_wandb(self, stats, step=None):
        if not self.__if_log_wandb:
            return
        if self.__wandb_run is None:
            wandb.login()
            for _ in range(10):
                try:
                    self.__wandb_run = wandb.init(dir=base.cluster.get_user_tmp(),
                                                  resume="allow",
                                                  **self.__wandb_args)
                    break
                except wandb.errors.UsageError as e:
                    time.sleep(5)
            else:
                raise e

        self.__wandb_run.log(stats, step=step)

    def wandb_resumed(self):
        if self.__wandb_run:
            return self.__wandb_run.resumed
        else:
            return False


class WorkerResourceMonitor:

    def __init__(self):
        self.__pid = os.getpid()
        self.__this_process = psutil.Process(self.__pid)
        self.popen_gpu_percent = None
        self.popen_gpu_mem = None

    def get_pid(self):
        """ get pid of current worker"""
        return self.__pid

    def cpu_percent(self):
        return self.__this_process.cpu_percent()

    def memory(self):
        mem_info = self.__this_process.memory_full_info()
        if platform.system() in ["Darwin", "Windows"]:
            return mem_info.rss, mem_info.vms, 0
        else:
            return mem_info.rss, mem_info.vms, mem_info.shared

    def __execute_gpu_percent(self):
        cmd = "nvidia-smi pmon -c 1".split(" ")
        try:
            self.popen_gpu_percent = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        except FileNotFoundError:
            return

    def gpu_percent(self):
        try:
            if not self.popen_gpu_percent:
                return 0, 0
            if self.popen_gpu_percent.returncode:
                return 0, 0
            out = self.popen_gpu_percent.stdout.readlines()
            lines = [l.decode('utf-8') for l in out]
            for l in lines:
                words = l.split(" ")
                while "" in words:
                    words.remove("")
                if len(words) == 0:
                    continue
                if not words[0].isdigit():
                    continue
                ppid, sm, mem = int(words[1]), words[3], words[4]
                sm = 0 if sm == '-' else int(sm)
                mem = 0 if mem == '-' else int(mem)
                if ppid == self.__pid:
                    return sm, mem
            return 0, 0
        except:
            return 0, 0
        finally:
            self.__execute_gpu_percent()

    def __execute_gpu_mem(self):
        cmd = "nvidia-smi pmon -s m -c 1".split(" ")
        try:
            self.popen_gpu_mem = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        except FileNotFoundError:
            return

    def gpu_mem(self):
        try:
            if self.popen_gpu_mem.returncode:
                return 0
            out = self.popen_gpu_mem.stdout.readlines()
            lines = [l.decode('utf-8') for l in out]
            for l in lines:
                words = l.split(" ")
                while "" in words:
                    words.remove("")
                if len(words) == 0:
                    continue
                if not words[0].isdigit():
                    continue
                ppid, mem = int(words[1]), words[3]
                mem = 0 if mem == '-' else int(mem)
                if ppid == self.__pid:
                    return mem
            return 0
        except:
            return 0
        finally:
            self.__execute_gpu_mem()

    def thread_profiles(self):
        thread_profile = self.__this_process.threads()
        return thread_profile


_MONITOR_THREAD_LOG_FREQUENCY_SECONDS = 10
_MONITOR_THREAD_SLEEP_INTERVAL_SECONDS = 3


class MonitorThread(threading.Thread):

    def __init__(self, monitor: Monitor):
        super().__init__()
        self.__monitor = monitor
        self.__resource_monitor = WorkerResourceMonitor()
        self.__stats = dict()
        self.__last_log_time_ns = None

    def update_stats(self, new_stats):
        self.__stats.update(new_stats)

    def thread_profiles(self):
        profiles = self.__resource_monitor.thread_profiles()
        thread_id = self.native_id
        all_time = 0
        this_thread_time = 0
        for p in profiles:
            all_time += p.user_time + p.system_time
            if thread_id == p.id:
                this_thread_time += p.user_time + p.system_time
        return this_thread_time, all_time, this_thread_time / all_time

    def __log(self):
        cpu_percent = self.__resource_monitor.cpu_percent()
        memory_rss, memory_vms, memory_shared = self.__resource_monitor.memory()
        gpu_percent, gpu_mem_util = self.__resource_monitor.gpu_percent()
        gpu_mem_used = self.__resource_monitor.gpu_mem()
        self.__monitor.metric("marl_worker_cpu_percent").observe(cpu_percent)
        self.__monitor.metric("marl_worker_memory_rss_mb").observe(memory_rss / (1024**2))
        self.__monitor.metric("marl_worker_memory_vms_mb").observe(memory_vms / (1024**2))
        self.__monitor.metric("marl_worker_memory_shared_mb").observe(memory_shared / (1024**2))
        self.__monitor.metric("marl_worker_gpu_percent").observe(gpu_percent)
        self.__monitor.metric("marl_worker_gpu_mem_util_percent").observe(gpu_mem_util)
        self.__monitor.metric("marl_worker_gpu_memory_mb").observe(gpu_mem_used)

        # log other stats from worker base
        self.__monitor.log_wandb(self.__stats)
        self.__stats = {}

    def print(self):
        # for debug in workers
        cpu_percent = self.__resource_monitor.cpu_percent()
        memory_rss, memory_vms, memory_shared = self.__resource_monitor.memory()
        gpu_percent, gpu_mem_util = self.__resource_monitor.gpu_percent()
        gpu_mem_used = self.__resource_monitor.gpu_mem()
        info_string = f"cpu_percent: {cpu_percent}, memory_rss: {memory_rss}, memory_vms: {memory_vms}, memory_shared: {memory_shared}, gpu_percent: {gpu_percent}, gpu_mem_util: {gpu_mem_util}, gpu_mem_used: {gpu_mem_used}"
        logger.info(info_string)

    def run(self):
        while True:
            time.sleep(_MONITOR_THREAD_SLEEP_INTERVAL_SECONDS)
            now = time.monotonic_ns()
            if self.__last_log_time_ns is not None:  # Log with a frequency.
                if (now - self.__last_log_time_ns) / 1e9 < _MONITOR_THREAD_LOG_FREQUENCY_SECONDS:
                    continue
            self.__last_log_time_ns = now
            self.__log()


class DummyPrometheusMetrics:
    """Dummy Prometheus Metrics for passing tests."""

    def __init__(self):
        pass

    def observe(self, val):
        pass

    def inc(self, val):
        pass

    def dec(self, val):
        pass

    def set(self, val):
        pass

    class time:

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass


class DummyMonitor:
    """ Dummy Monitor for passing tests.
    """

    def __init__(self, metrics: MonitorInfo):
        pass

    def update_metrics(self, metric_specs: Dict):
        pass

    def metric(self, metric_name: str):
        return DummyPrometheusMetrics()

    def new_wandb_run(self, new_wandb_args):
        pass

    def log_wandb(self, stats, step=None):
        pass

    def wandb_resumed(self):
        pass


class DummyMonitorThread(threading.Thread):

    def __init__(self):
        pass

    def update_stats(self, new_stats):
        pass
