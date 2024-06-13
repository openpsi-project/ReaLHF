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
  return cost_table[key];
}

void RPCInstance::resolve_parameter_sync(std::vector<RPCInstance *> tmp_graph,
                                         std::unordered_map<std::string, uint64_t> &cost_table) {
  // add parameter synchronization edges
  if (!param_sync) return;

  // dst to train
  uint64_t from_cost =
      parameter_sync_cost(param_sync_size, param_sync_rpc_exe_ptr, rpc_exe_ptr, cost_table);
  uint64_t to_cost =
      parameter_sync_cost(param_sync_size, rpc_exe_ptr, param_sync_rpc_exe_ptr, cost_table);
  // if (param_sync_cost > 0)
  //     std::cout << "Param sync cost " << param_sync_cost << " from "
  //             << param_sync_rpc_exe_ptr -> rpc_ptr -> rpc_name << " to "
  //             << rpc_exe_ptr -> rpc_ptr -> rpc_name << std::endl;

  // add param sync from src to dst
  RPCExecution *from_src_exe =
      new RPCExecution(rpc_ptr, param_sync_rpc_exe_ptr->device_mesh,
                       param_sync_rpc_exe_ptr->model_parallel_strategy, from_cost, 0, 0);
  RPCInstance *from_src = new RPCInstance(rpc_ptr, id, name + ":from_src");
  from_src->rpc_exe_ptr = from_src_exe;

  RPCExecution *from_dst_exe = new RPCExecution(
      rpc_ptr, rpc_exe_ptr->device_mesh, rpc_exe_ptr->model_parallel_strategy, from_cost, 0, 0);
  RPCInstance *from_dst = new RPCInstance(rpc_ptr, id, name + ":from_dst");
  from_dst->rpc_exe_ptr = from_dst_exe;

  // bool overlap = src -> rpc_exe_ptr -> device_mesh.overlap(
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
      new RPCExecution(rpc_ptr, param_sync_rpc_exe_ptr->device_mesh,
                       param_sync_rpc_exe_ptr->model_parallel_strategy, to_cost, 0, 0);
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
  return rpc_ptr->rpc_name + " on " + device_mesh.name
         + ", parallel strategy: " + model_parallel_strategy.to_string();
};