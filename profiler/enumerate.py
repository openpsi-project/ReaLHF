from profiler.device_mesh import *
from profiler.estimate import *
from profiler.experiments import *

from impl.model.nn.flash_mqat.flash_mqat_base import FlashMQATConfig

GPU_MEM_CAP = 80 * (1024**3)


@dataclasses.dataclass
class RPCExecution:
    rpc: ModelRPC
    device_mesh: DeviceMesh
    parallel_strategy: ModelParallelStrategy
    time_cost: float = None
    mem: float = None
    static_mem: float = None
    rpc_name: str = None

    def __post_init__(self):
        self.rpc_name = model_rpc_name(self.rpc)

    def __hash__(self):
        return hash((self.rpc_name, self.device_mesh, self.parallel_strategy))


class GroupedRPCExecutions:

    def __init__(self):
        self.rpc_exe_groups: List[List[RPCExecution]] = []
        self.rpc_exe_group_models: List[List[str]] = []
        self.mem_static: List[List[int]] = []
        self.mem_active: List[List[int]] = []
        self.mem_costs: List[int] = []

    def add(self, rpc_exe: RPCExecution):
        overlap_flag = False
        for i, group in enumerate(self.rpc_exe_groups):
            device_meshes = [x.device_mesh for x in group]
            if is_all_overlap(device_meshes, rpc_exe.device_mesh):
                group.append(rpc_exe)
                if rpc_exe.rpc.model_name not in self.rpc_exe_group_models[i]:
                    self.rpc_exe_group_models[i].append(rpc_exe.rpc.model_name)
                    self.mem_static[i].append(rpc_exe.mem)
                    self.mem_active[i].append(rpc_exe.mem - rpc_exe.static_mem)
                else:
                    j = self.rpc_exe_group_models[i].index(rpc_exe.rpc.model_name)
                    self.mem_static[i][j] = max(self.mem_static[i][j], rpc_exe.mem)
                    self.mem_active[i][j] = max(self.mem_active[i][j], rpc_exe.mem - rpc_exe.static_mem)
                self.mem_costs[i] = sum(self.mem_static[i]) + max(self.mem_active[i])
                overlap_flag = True

        if not overlap_flag:
            self.rpc_exe_groups.append([rpc_exe])
            self.rpc_exe_group_models.append([rpc_exe.rpc.model_name])
            self.mem_static.append([rpc_exe.mem])
            self.mem_active.append([rpc_exe.mem - rpc_exe.static_mem])
            self.mem_costs.append(rpc_exe.mem)

    def total_mem_cost(self):
        return sum(self.mem_costs)


def estimate_function_call_memory(interface_type: ModelInterfaceType,
                                  batch_size: int,
                                  seq_len: int,
                                  model_config: FlashMQATConfig,
                                  parallel_strategy: ModelParallelStrategy,
                                  gradient_checkpointing: bool = False):
    h = model_config.hidden_dim
    i = model_config.intermediate_dim
    v = model_config.vocab_size
    s = seq_len
    b = batch_size
    L = model_config.n_layers
    # for llama actor only
    n_params = 3 * v * h + (3 * h * i + 4 * h * h) * L
    param_mem = 2 * n_params
    grad_mem = 2 * n_params
    optimizer_mem = 16 * n_params

    num_pp = parallel_strategy.num_pp
    num_mp = parallel_strategy.num_mp
    num_dp = parallel_strategy.num_dp
    # print(f"Parallel strategy: num_pp: {num_pp}, num_mp: {num_mp}, num_dp: {num_dp}")
    # zero1, pp and mp divide evenly
    # enable sequence parallel
    if interface_type == ModelInterfaceType.TRAIN_STEP:
        static_mem = (param_mem + grad_mem) // (num_pp * num_mp) +\
                     optimizer_mem // (num_pp * num_dp * num_mp)
        micro_bs = b // (2 * num_pp * num_dp)
        active_mem = (micro_bs * s * h * num_pp * 2) * 2 * L // (num_pp * num_mp)
        # enabled gradient ckpt
        # print(f"train static_mem: {static_mem/(1024*1024*1024):02f} GB, "
        #       f"active_mem: {active_mem/(1024*1024*1024):02f} GB, "
        #       f"total: {(static_mem + active_mem)/(1024*1024*1024):02f} GB")
        return static_mem + active_mem, static_mem
    elif interface_type == ModelInterfaceType.INFERENCE:
        static_mem = param_mem // (num_pp * num_mp)
        active_mem = 0  # no tensor need to be stored in inference
        # print(f"inference static_mem: {static_mem/(1024*1024*1024):02f} GB")
        return static_mem, static_mem
    elif interface_type == ModelInterfaceType.GENERATE:
        static_mem = param_mem // (num_pp * num_mp)
        active_mem = 2 * (2 * b * s * h) * L // (num_pp * num_mp * num_dp)  # kv cache
        # print(f"generate static_mem: {static_mem/(1024*1024*1024):02f} GB, "
        #       f"kv_cache_mem: {kv_cache_mem/(1024*1024*1024):02f} GB, "
        #       f"total: {(static_mem + kv_cache_mem)/(1024*1024*1024):02f} GB")
        return static_mem + active_mem, static_mem


def enumerate_rpc_executions(exp: ProfileExperiment, rpc: ModelRPC, device_mesh: DeviceMesh,
                             model_config: FlashMQATConfig) -> List[RPCExecution]:
    sub_device_meshes = find_sub_device_meshes(device_mesh)
    feasible = []
    for sub_device_mesh in sub_device_meshes:
        ps = find_parallel_strategies(sub_device_mesh)
        for p in ps:
            mem_cost, static_mem = estimate_function_call_memory(rpc.interface_type, rpc.min_n_seqs,
                                                                 rpc.max_n_tokens // rpc.min_n_seqs,
                                                                 model_config, p)
            time_cost = estimate_rpc_cost(exp, rpc, model_config, p)
            if mem_cost * 1.2 < GPU_MEM_CAP:
                feasible.append(RPCExecution(rpc, sub_device_mesh, p, time_cost, mem_cost, static_mem))
    return feasible


def enumerate_model_device_mappings(exp: ProfileExperiment):
    device_mesh = make_device_mesh_from_name(exp.device_mesh_name)
    rpc_exe_table = {}
    avg_time_cost = []
    min_time_cost_sum = 0

    for rpc in exp.model_rpcs:
        model_type = exp.model_names_to_types[rpc.model_name]
        flash_mqat_config = load_model_config(exp.model_paths[exp.model_types.index(model_type)])
        feasible = enumerate_rpc_executions(exp, rpc, device_mesh, flash_mqat_config)
        print(f"{model_rpc_name(rpc)} feasible: {len(feasible)}")
        feasible.sort(key=lambda x: x.time_cost)
        # feasible = feasible[:10]

        for rpc_exe in feasible[:10]:
            rpc_exe: RPCExecution
            print(f"time_cost: {rpc_exe.time_cost/(1e3)} ms, "
                  f"sub_device_mesh: {rpc_exe.device_mesh}, "
                  f"parallel_strategy: {rpc_exe.parallel_strategy}, "
                  f"mem_cost: {rpc_exe.mem/(1024*1024*1024):02f} GB, "
                  f"static_mem_cost: {rpc_exe.static_mem/(1024*1024*1024):02f} GB")

        rpc_exe_table[model_rpc_name(rpc)] = feasible
        avg_time_cost.append((sum([x.time_cost for x in feasible][:10]) / 10, model_rpc_name(rpc)))
        min_time_cost_sum += feasible[0].time_cost

    avg_time_cost.sort(key=lambda x: x[0], reverse=True)
    sorted_model_rpc_names = [x[1] for x in avg_time_cost]
    print(sorted_model_rpc_names)
    count = 0
    inner_count = 0
    valid_count = 0
    index = [0 for _ in range(len(exp.model_rpcs))]
    # prune by memory
    while True:
        grouped_rpc_exe = GroupedRPCExecutions()
        time_cost_sum = 0
        for i in range(len(exp.model_rpcs)):
            rpc_name = sorted_model_rpc_names[i]
            rpc_exe: RPCExecution = rpc_exe_table[rpc_name][index[i]]
            time_cost_sum += rpc_exe.time_cost
            if time_cost_sum > 1.5 * min_time_cost_sum:
                break
            # check overlap situation
            grouped_rpc_exe.add(rpc_exe)
            inner_count += 1
            current_mem = grouped_rpc_exe.total_mem_cost()
            # if count % 10 == 0:
            #     print(f"{count} {index} {i} {current_mem/(1024*1024*1024):02f} GB")
            if current_mem > GPU_MEM_CAP * 1.2:  # offload space
                break
        else:
            valid_count += 1
            print(f"{index} {current_mem/(1024*1024*1024):02f} GB")

        outer_loop_break_flag = True
        while i >= 0:
            if index[i] + 1 < len(rpc_exe_table[sorted_model_rpc_names[i]]):
                index[i] += 1
                outer_loop_break_flag = False
                break
            else:
                i -= 1

        for j in range(i + 1, len(exp.model_rpcs)):
            index[j] = 0

        count += 1
        if outer_loop_break_flag:
            print(f"index {index}, i {i}")
            break

    print(valid_count)

    # train and share back bone
    # time_cost_list = []
    # model_type = exp.model_names_to_types[actor_train.model_name]
    # flash_mqat_config = load_model_config(exp.model_paths[exp.model_types.index(model_type)])
    # feasible = enumerate_rpc(actor_train, device_mesh, flash_mqat_config)
    # print(f"{model_rpc_name(actor_train)} feasible: {len(feasible)}")

    # for sub_device_mesh, p, mem_cost in feasible:
    #     time_cost = estimate_rpc_cost(exp, actor_train, flash_mqat_config, p)
    #     time_cost_list.append((time_cost, sub_device_mesh, p, mem_cost))

    # time_cost_list.sort(key=lambda x: x[0])
    # for t, s, p, mem_cost in time_cost_list[:10]:
    #     print(f"time_cost: {t}, sub_device_mesh: {s}, parallel_strategy: {p}, "
    #           f"mem_cost: {mem_cost/(1024*1024*1024):02f} GB")
    #     mem_cost =  estimate_function_call_memory(actor_generate.interface_type,
    #                                               actor_generate.min_n_seqs,
    #                                               actor_generate.max_n_tokens // actor_generate.min_n_seqs,
    #                                               flash_mqat_config,
    #                                               p)
    #     time_cost = estimate_rpc_cost(exp, actor_generate, flash_mqat_config, p)
    #     print(f"share backbone generate: time_cost {time_cost} "
    #           f"mem_cost {mem_cost/(1024*1024*1024):02f}")

    # # generate and share backbone
    # time_cost_list = []
    # model_type = exp.model_names_to_types[actor_generate.model_name]
    # flash_mqat_config = load_model_config(exp.model_paths[exp.model_types.index(model_type)])
    # feasible = enumerate_rpc(actor_generate, device_mesh, flash_mqat_config)
    # print(f"{model_rpc_name(actor_generate)} feasible: {len(feasible)}")

    # for sub_device_mesh, p, mem_cost in feasible:
    #     time_cost = estimate_rpc_cost(exp, actor_generate, flash_mqat_config, p)
    #     time_cost_list.append((time_cost, sub_device_mesh, p, mem_cost))

    # time_cost_list.sort(key=lambda x: x[0])
    # for t, s, p, mem_cost in time_cost_list[:10]:
    #     print(f"time_cost: {t}, sub_device_mesh: {s}, parallel_strategy: {p}, "
    #           f"mem_cost: {mem_cost/(1024*1024*1024):02f} GB")
    #     mem_cost =  estimate_function_call_memory(actor_train.interface_type,
    #                                               actor_train.min_n_seqs,
    #                                               actor_train.max_n_tokens // actor_train.min_n_seqs,
    #                                               flash_mqat_config,
    #                                               p)
    #     time_cost = estimate_rpc_cost(exp, actor_train, flash_mqat_config, p)
    #     print(f"share backbone generate: time_cost {time_cost}, "
    #           f"mem_cost {mem_cost/(1024*1024*1024):02f} GB")


if __name__ == "__main__":
    exp = ProfileExperiment()
    enumerate_model_device_mappings(exp)
