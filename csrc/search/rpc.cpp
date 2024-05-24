#include <rpc.hpp>
#include <device_mesh.hpp>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <iomanip>

RPC::RPC(std::string model_name, std::string rpc_name, std::string interface_type)
    : model_name(model_name), rpc_name(rpc_name), interface_type(interface_type) {};

RPCExecution::RPCExecution(RPC *rpc_ptr, DeviceMesh &device_mesh,
                           ModelParallelStrategy &model_parallel_strategy, uint64_t time_cost,
                           uint64_t mem, uint64_t static_mem)
    : rpc_ptr(rpc_ptr),
      device_mesh(device_mesh),
      model_parallel_strategy(model_parallel_strategy),
      time_cost(time_cost),
      mem(mem),
      static_mem(static_mem) {};

bool OverlapGroup::maybe_add(RPCExecution *rpc_exe) {
  if (rpc_executions.empty()) {
    rpc_executions.insert(rpc_exe);
    device_meshes.insert(&rpc_exe->device_mesh);
    mem_static = rpc_exe->static_mem;
    mem_active = rpc_exe->mem - rpc_exe->static_mem;
    return true;
  }
  if (is_all_overlap(device_meshes, rpc_exe->device_mesh)) {
    rpc_executions.insert(rpc_exe);
    // bool dm_in_group = device_meshes.find(&rpc_exe -> device_mesh) != device_meshes.end();
    device_meshes.insert(&rpc_exe->device_mesh);
    mem_static += rpc_exe->static_mem;
    mem_active = std::max(mem_active, rpc_exe->mem - rpc_exe->static_mem);
    return true;
  }
  return false;
};

void DeviceMeshGroup::add_to_groups(RPCExecution *rpc_exe) {
  if (overlap_groups.empty()) {
    OverlapGroup *og = new OverlapGroup();
    og->maybe_add(rpc_exe);
    overlap_groups.push_back(og);
    return;
  }

  std::vector<OverlapGroup *> tmp_new_ogs;
  for (OverlapGroup *og : overlap_groups) {
    // OverlapGroup og_copy = *og;
    bool update = og->maybe_add(rpc_exe);
    if (!update) {
      tmp_new_ogs.push_back(new OverlapGroup());
      tmp_new_ogs.back()->maybe_add(rpc_exe);
      for (RPCExecution *rpc_exe : og->rpc_executions) { tmp_new_ogs.back()->maybe_add(rpc_exe); }
    }
    tmp_new_ogs.push_back(og);
  }
  overlap_groups.clear();
  for (OverlapGroup *og : tmp_new_ogs) { overlap_groups.push_back(og); }
};

void GroupedRPCExecutions::resolve(RPCExecution *rpc_exe) {}

void GroupedRPCExecutions::add(RPCExecution *rpc_exe) { group.add_to_groups(rpc_exe); };

void GroupedRPCExecutions::offload(std::string model_name) {};

uint64_t GroupedRPCExecutions::total_mem_cost() {
  uint64_t max_mem = 0;
  for (auto &og : group.overlap_groups) {
    // double og_ma = (og -> mem_active/(1024*1024))/1024.0;
    // double og_ms = (og -> mem_static/(1024*1024))/1024.0;
    // std::cout << "og size " << og -> rpc_executions.size()
    //           << " mem active " << og_ma << " GB"
    //           << " mem static " << og_ms << " GB" << std::endl;
    if (og->mem_active + og->mem_static > max_mem) { max_mem = og->mem_active + og->mem_static; }
  }
  return max_mem;
};

RPCInstance::RPCInstance(RPC *rpc_ptr, int id, std::string name)
    : rpc_ptr(rpc_ptr), id(id), name(name) {};

void RPCInstance::remove_parent(RPCInstance *parent) {
  auto it = std::find(parents.begin(), parents.end(), parent);
  if (it != parents.end()) { parents.erase(it); }
};

void RPCInstance::remove_child(RPCInstance *child) {
  auto it = std::find(children.begin(), children.end(), child);
  if (it != children.end()) { children.erase(it); }
};

void RPCInstance::add_parent(RPCInstance *parent) { parents.push_back(parent); };

void RPCInstance::add_child(RPCInstance *child) { children.push_back(child); };

void RPCInstance::remove_tmp_child(RPCInstance *child) {
  auto it = std::find(tmp_children.begin(), tmp_children.end(), child);
  if (it != tmp_children.end()) { tmp_children.erase(it); }
};

void RPCInstance::remove_tmp_parent(RPCInstance *parent) {
  auto it = std::find(tmp_parents.begin(), tmp_parents.end(), parent);
  if (it != tmp_parents.end()) { tmp_parents.erase(it); }
};

void RPCInstance::add_tmp_parent(RPCInstance *parent) { tmp_parents.push_back(parent); };

void RPCInstance::add_tmp_child(RPCInstance *child) { tmp_children.push_back(child); };

uint64_t parameter_sync_cost(uint64_t model_size, RPCExecution *src, RPCExecution *dst,
                             std::unordered_map<std::string, uint64_t> &cost_table) {
  // 7b size 13738442752 Bytes
  // double size_multiplier = double(model_size) / 13738442752.0;
  std::string model_key = std::to_string(model_size);
  std::string src_key = src->model_parallel_strategy.to_key();
  std::string dst_key = dst->model_parallel_strategy.to_key();
  if (src_key == dst_key) return 0;
  std::string key = model_key + "," + src_key + "," + dst_key;
  if (cost_table.find(key) == cost_table.end()) {
    // std::cout << "key " << key << " not found" << std::endl;
    return 0;
  }
  // uint64_t res = static_cast<uint64_t>(cost_table[key] * size_multiplier);
  // std::cout << "key " << key << " " << cost_table[key] << std::endl;
  return cost_table[key];

  // return 0;
  // std::cout << "param sync cost: " << src -> rpc_ptr -> rpc_name << " to "
  //           << dst -> rpc_ptr -> rpc_name << " ";

  // if (src -> device_mesh == dst -> device_mesh
  //     && src -> model_parallel_strategy == dst -> model_parallel_strategy) {
  //     // std::cout << 0 << std::endl;
  //     return uint64_t(0);
  // } else {
  //     uint64_t total_param_size = param_size_bytes *
  //                                 src -> model_parallel_strategy.num_dp;
  //     // uint64_t src_size_bytes_per_host = total_param_size / src -> device_mesh.n_nodes;
  //     // uint64_t dst_size_bytes_per_host = total_param_size / dst -> device_mesh.n_nodes;
  //     // std::cout << std::max(src_size_bytes_per_host,
  //     //                      dst_size_bytes_per_host) / comm_stats.remote_send << std::endl;

  //     // uint64_t host_size = std::max(src_size_bytes_per_host,
  //     //                               dst_size_bytes_per_host);

  //     uint64_t remote_size = remote_param_realloc_size(total_param_size, src, dst);
  //     std::cout  << " remote_size " << remote_size << "cost" << remote_size/20000 << std::endl;
  //     return remote_size/20000;
  // }
  // return uint64_t(0);
}

uint64_t remote_param_realloc_size(uint64_t size, RPCExecution *src, RPCExecution *dst) {
  ModelParallelStrategy src_p = src->model_parallel_strategy;
  ModelParallelStrategy dst_p = dst->model_parallel_strategy;
  // std::cout << "src device_mesh " << src -> device_mesh.device_mesh_name
  //           << " src parallel strategy " << src_p.to_string() << std::endl;

  // std::cout << "dst device_mesh " << dst -> device_mesh.device_mesh_name
  //           << " dst parallel strategy " << dst_p.to_string() << std::endl;
  int src_pp = src_p.num_pp;
  int dst_pp = dst_p.num_pp;
  std::vector<std::unordered_set<std::string>> src_node_groups = divide(src->device_mesh, src_pp);
  std::vector<std::unordered_set<std::string>> dst_node_groups = divide(dst->device_mesh, dst_pp);
  std::unordered_map<std::string, int> node_remote_layer;

  int local_layer = 0;
  int max_remote_layers = 0;
  int n_layers = src_pp * dst_pp;
  std::vector<std::pair<int, int>> layer_pairs;

  for (int i = 0; i < n_layers; i++) {
    int left = i / dst_pp, right = i / src_pp;
    // std::cout << "num_layers " << n_layers << std::endl;
    // std::cout << "left " << left << " right " << right << std::endl;
    layer_pairs.push_back({left, right});
  }

  for (auto &p : layer_pairs) {
    // std::cout << "p " << p.first << " " << p.second << std::endl;
    // std::cout << "src node group size " << src_node_groups.size() << std::endl;
    // std::cout << "dst node group size " << dst_node_groups.size() << std::endl;
    std::unordered_set<std::string> src_node_group = src_node_groups[p.first];
    std::unordered_set<std::string> dst_node_group = dst_node_groups[p.second];
    bool dst_in_src = true;
    for (std::string node_name : dst_node_group) {
      if (src_node_group.find(node_name) == src_node_group.end()) {
        dst_in_src = false;
        break;
      }
    }
    if (!dst_in_src) {
      for (std::string node_name : dst_node_group) {
        node_remote_layer[node_name] += 1;
        if (node_remote_layer[node_name] > max_remote_layers) {
          max_remote_layers = node_remote_layer[node_name];
        }
      }
      for (std::string node_name : src_node_group) {
        node_remote_layer[node_name] += 1;
        if (node_remote_layer[node_name] > max_remote_layers) {
          max_remote_layers = node_remote_layer[node_name];
        }
      }
    }
  }

  return (size * max_remote_layers) / n_layers;
}

void RPCInstance::resolve_parameter_sync(std::vector<RPCInstance *> tmp_graph,
                                         std::unordered_map<std::string, uint64_t> &cost_table) {
  // add parameter synchronization edges
  if (!param_realloc) return;

  // dst to train
  uint64_t from_cost =
      parameter_sync_cost(param_realloc_size, param_realloc_rpc_exe_ptr, rpc_exe_ptr, cost_table);
  uint64_t to_cost =
      parameter_sync_cost(param_realloc_size, rpc_exe_ptr, param_realloc_rpc_exe_ptr, cost_table);
  // if (param_realloc_cost > 0)
  //     std::cout << "Param sync cost " << param_realloc_cost << " from "
  //             << param_realloc_rpc_exe_ptr -> rpc_ptr -> rpc_name << " to "
  //             << rpc_exe_ptr -> rpc_ptr -> rpc_name << std::endl;

  // add param sync from src to dst
  RPCExecution *from_src_exe =
      new RPCExecution(rpc_ptr, param_realloc_rpc_exe_ptr->device_mesh,
                       param_realloc_rpc_exe_ptr->model_parallel_strategy, from_cost, 0, 0);
  RPCInstance *from_src = new RPCInstance(rpc_ptr, id, name + ":from_src");
  from_src->rpc_exe_ptr = from_src_exe;

  RPCExecution *from_dst_exe = new RPCExecution(
      rpc_ptr, rpc_exe_ptr->device_mesh, rpc_exe_ptr->model_parallel_strategy, from_cost, 0, 0);
  RPCInstance *from_dst = new RPCInstance(rpc_ptr, id, name + ":from_dst");
  from_dst->rpc_exe_ptr = from_dst_exe;

  // bool overlap = src -> rpc_exe_ptr -> device_mesh.is_overlap(
  //                 dst -> rpc_exe_ptr -> device_mesh);

  for (RPCInstance *parent : parents) {
    parent->remove_tmp_child(this);
    from_src->add_tmp_parent(parent);
    from_dst->add_tmp_parent(parent);
    parent->add_tmp_child(from_src);
    parent->add_tmp_child(from_dst);
  }
  this->tmp_parents.clear();

  from_src->add_tmp_child(this);
  from_dst->add_tmp_child(this);
  this->add_tmp_parent(from_src);
  this->add_tmp_parent(from_dst);

  tmp_graph.push_back(from_src);
  tmp_graph.push_back(from_dst);

  tmp_ris.push_back(from_src);
  tmp_ris.push_back(from_dst);
  tmp_exes.push_back(from_src_exe);
  tmp_exes.push_back(from_dst_exe);

  // add param sync from dst to src
  RPCExecution *to_src_exe = new RPCExecution(rpc_ptr, rpc_exe_ptr->device_mesh,
                                              rpc_exe_ptr->model_parallel_strategy, to_cost, 0, 0);
  RPCInstance *to_src = new RPCInstance(rpc_ptr, id, name + ":to_src");
  to_src->rpc_exe_ptr = to_src_exe;

  RPCExecution *to_dst_exe =
      new RPCExecution(rpc_ptr, param_realloc_rpc_exe_ptr->device_mesh,
                       param_realloc_rpc_exe_ptr->model_parallel_strategy, to_cost, 0, 0);
  RPCInstance *to_dst = new RPCInstance(rpc_ptr, id, name + ":to_dst");
  to_dst->rpc_exe_ptr = to_dst_exe;

  for (RPCInstance *child : children) {
    child->remove_tmp_parent(this);
    to_src->add_tmp_child(child);
    to_dst->add_tmp_child(child);
    child->add_tmp_parent(to_src);
    child->add_tmp_parent(to_dst);
  }
  this->tmp_children.clear();

  to_src->add_tmp_parent(this);
  to_dst->add_tmp_parent(this);
  this->add_tmp_child(to_src);
  this->add_tmp_child(to_dst);

  tmp_graph.push_back(to_src);
  tmp_graph.push_back(to_dst);

  tmp_ris.push_back(to_src);
  tmp_ris.push_back(to_dst);
  tmp_exes.push_back(to_src_exe);
  tmp_exes.push_back(to_dst_exe);
}

// void RPCInstance::resolve_offload(std::vector<RPCInstance*> tmp_graph,
//                                   CommStats& comm_stats) {
//     // add offload edges
//     if (!offload) return;
//     uint64_t offload_store_cost = offload_size / comm_stats.offload_store;
//     uint64_t offload_load_cost = offload_size / comm_stats.offload_load;
//     // std::cout << "model " << rpc_ptr -> model_name << " offload store cost "
//     //           << offload_store_cost << " offload load cost " << offload_load_cost <<
//     std::endl;

//     // if (offload_store_cost > 0)
//     //     std::cout << "Offload store cost " << offload_store_cost
//     //               << "; Offload load cost " << offload_load_cost
//     //               << "; for rpc instance " << name << std::endl;

//     RPCExecution* store_exe = new RPCExecution(rpc_ptr,
//                                                rpc_exe_ptr -> device_mesh,
//                                                rpc_exe_ptr -> model_parallel_strategy,
//                                                offload_store_cost, 0, 0);
//     RPCInstance* store = new RPCInstance(rpc_ptr, id, name + ":store");
//     store -> rpc_exe_ptr = store_exe;

//     RPCExecution* load_exe = new RPCExecution(rpc_ptr,
//                                               rpc_exe_ptr -> device_mesh,
//                                               rpc_exe_ptr -> model_parallel_strategy,
//                                               offload_load_cost, 0, 0);
//     RPCInstance* load = new RPCInstance(rpc_ptr, id, name + ":load");
//     load -> rpc_exe_ptr = load_exe;

//     // add load to graph
//     for (RPCInstance* parent : parents) {
//         parent -> remove_tmp_child(this);
//         load -> add_tmp_parent(parent);
//         parent -> add_tmp_child(load);
//     }
//     this -> tmp_parents.clear();

//     load -> add_tmp_child(this);
//     this -> add_tmp_parent(load);

//     // add store to graph
//     for (RPCInstance* child : children) {
//         child -> remove_tmp_parent(this);
//         store -> add_tmp_child(child);
//         child -> add_tmp_parent(store);
//     }
//     this -> tmp_children.clear();

//     store -> add_tmp_parent(this);
//     this -> add_tmp_child(store);

//     tmp_graph.push_back(store);
//     tmp_graph.push_back(load);

//     tmp_ris.push_back(store);
//     tmp_ris.push_back(load);
//     tmp_exes.push_back(store_exe);
//     tmp_exes.push_back(load_exe);
// }

CommStats::CommStats(uint64_t local_send, uint64_t local_recv, uint64_t remote_send,
                     uint64_t remote_recv, uint64_t offload_store, uint64_t offload_load)
    : local_send(local_send),
      local_recv(local_recv),
      remote_send(remote_send),
      remote_recv(remote_recv),
      offload_store(offload_store),
      offload_load(offload_load) {};

// ModelConfig::ModelConfig(std::string model_name,
//                          uint64_t param_size_bytes)
// : model_name(model_name), param_size_bytes(param_size_bytes) {
// };

std::string RPCExecution::to_string() {
  return rpc_ptr->rpc_name + " on " + device_mesh.device_mesh_name
         + ", parallel strategy: " + model_parallel_strategy.to_string();
};