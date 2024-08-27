# This file standardizes the name-resolve names used by different components of the system.
import getpass

USER_NAMESPACE = getpass.getuser()


def registry_root(user):
    return f"trial_registry/{user}"


def trial_registry(experiment_name, trial_name):
    return f"trial_registry/{USER_NAMESPACE}/{experiment_name}/{trial_name}"


def trial_root(experiment_name, trial_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}"


def worker_status(experiment_name, trial_name, worker_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/status/{worker_name}"


def worker_root(experiment_name, trial_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/worker/"


def worker(experiment_name, trial_name, worker_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/worker/{worker_name}"


def worker_key(experiment_name, trial_name, key):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/worker_key/{key}"


def request_reply_stream(experiment_name, trial_name, stream_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/request_reply_stream/{stream_name}"


def request_reply_stream_root(experiment_name, trial_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/request_reply_stream/"


def distributed_root(experiment_name, trial_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/distributed/"


def distributed_peer(experiment_name, trial_name, model_name):
    return (
        f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/distributed/peer/{model_name}"
    )


def distributed_local_peer(experiment_name, trial_name, host_name, model_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/distributed/local_peer/{host_name}/{model_name}"


def distributed_master(experiment_name, trial_name, model_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/distributed/master/{model_name}"
