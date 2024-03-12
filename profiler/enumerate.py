from profiler.device_mesh import *
from profiler.estimate import load_model_config
from profiler.experiments import *

from impl.model.nn.flash_mqat.flash_mqat_base import FlashMQATConfig


def estimate_memory_usage(exp: ProfileExperiment):
    for rpc in exp.model_rpcs:
        interface_type = rpc.interface_type
        batch_size = rpc.min_n_seqs
        seq_len = rpc.max_n_tokens // batch_size
        model_type = exp.model_names_to_types[rpc.model_name]
        model_path = exp.model_paths[exp.model_names.index(rpc.model_name)]
        flash_mqat_config = load_model_config(model_path)


def estimate_function_call_memory(interface_type: ModelInterfaceType,
                                  batch_size: int,
                                  seq_len: int,
                                  model_config: FlashMQATConfig,
                                  parallel_strategy: ModelParallelStrategy,
                                  use_gradient_checkpointing: bool = False):
    pass


def enumerate_model_device_mappings(exp: ProfileExperiment):
    model_rpc_names = [model_rpc_name(rpc) for rpc in exp.model_rpcs]
    device_mesh = make_device_mesh_from_name(exp.device_mesh_name)
    sub_device_meshes = find_sub_device_meshes(device_mesh)

    # model_device_mapping = ModelDeviceMapping(
    #     model_names=exp.model_names,
    #     model_rpc_names=model_rpc_names,
    #     model_rpc_mapping=dict(zip(model_rpc_names, exp.model_rpcs)),
    #     model_device_mapping={f"{model}_rpc": device_mesh for model in models}
    # )
