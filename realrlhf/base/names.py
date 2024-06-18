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
    return (
        f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/status/{worker_name}"
    )


def worker_root(experiment_name, trial_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/worker/"


def worker(experiment_name, trial_name, worker_name):
    return (
        f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/worker/{worker_name}"
    )


def worker2(experiment_name, trial_name, worker_type, worker_index):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/worker/{worker_type}/{worker_index}"


def inference_stream(experiment_name, trial_name, stream_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/inference_stream/{stream_name}"


def inference_stream_constant(
    experiment_name, trial_name, stream_name, constant_name
):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/inference_stream_consts/{stream_name}/{constant_name}"


def sample_stream(experiment_name, trial_name, stream_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/sample_stream/{stream_name}"


def request_reply_stream(experiment_name, trial_name, stream_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/request_reply_stream/{stream_name}"


def trainer_ddp_peer(experiment_name, trial_name, model_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/trainer_ddp_peer/{model_name}"


def trainer_ddp_local_peer(experiment_name, trial_name, host_name, model_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/trainer_ddp_local_peer/{host_name}/{model_name}"


def trainer_ddp_master(experiment_name, trial_name, model_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/trainer_ddp_master/{model_name}"


def worker_key(experiment_name, trial_name, key):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/worker_key/{key}"


def parameter_subscription(experiment_name, trial_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/parameter_sub"


def parameter_server(experiment_name, trial_name, parameter_id_str):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/parameter_server/{parameter_id_str}"


def shared_memory(experiment_name, trial_name, stream_name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/shared_memory/{stream_name}"


def shared_memory_dock_server(
    experiment_name, trial_name, stream_name, server_type
):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/shared_memory_dock_server/{server_type}/{stream_name}"


def ray_cluster(experiment_name, trial_name, name):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/ray_cluster/{name}"


def model_controller(experiment_name, trial_name, model_name, dp_rank, mp_rank):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/model_controller/{model_name}/{dp_rank}/{mp_rank}"


def model_controller_barrier(
    experiment_name, trial_name, model_name, dp_rank, mp_rank
):
    return f"{USER_NAMESPACE}/{experiment_name}/{trial_name}/model_controller_barrier/{model_name}/{dp_rank}/{mp_rank}"
