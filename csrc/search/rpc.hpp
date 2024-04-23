#ifndef RPC_HPP
#define RPC_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <device_mesh.hpp>

class CommStats {
 public:
  uint64_t local_send, local_recv, remote_send, remote_recv, offload_store, offload_load;

  CommStats(uint64_t local_send, uint64_t local_recv, uint64_t remote_send, uint64_t remote_recv,
            uint64_t offload_store, uint64_t offload_load);
};

class RPC {
 public:
  std::string model_name;
  std::string rpc_name;
  // interface_type: 0=generate, 1=train_step, 2=inference
  std::string interface_type;

  RPC(std::string model_name, std::string rpc_name, std::string interface_type);
};

class RPCExecution {
 public:
  RPC *rpc_ptr;
  DeviceMesh &device_mesh;
  ModelParallelStrategy &model_parallel_strategy;
  uint64_t time_cost, mem, static_mem;

  RPCExecution(RPC *rpc_ptr, DeviceMesh &device_mesh,
               ModelParallelStrategy &model_parallel_strategy, uint64_t time_cost, uint64_t mem,
               uint64_t static_mem);

  std::string to_string();
};

class OverlapGroup {
 public:
  std::unordered_set<RPCExecution *> rpc_executions;
  std::unordered_set<DeviceMesh *> device_meshes;
  uint64_t mem_static;
  uint64_t mem_active;

  bool maybe_add(RPCExecution *rpc_exe);
};

class DeviceMeshGroup {
 public:
  // std::string device_mesh_name;
  std::vector<OverlapGroup *> overlap_groups;

  void add_to_groups(RPCExecution *rpc_exe);
};

class GroupedRPCExecutions {
 public:
  // std::unordered_map<std::string, DeviceMeshGroup> dn_to_group;
  DeviceMeshGroup group;

  void add(RPCExecution *rpc_exe);
  void resolve(RPCExecution *rpc_exe);
  void offload(std::string model_name);
  uint64_t total_mem_cost();
};

class RPCInstance {
 public:
  RPC *rpc_ptr;
  int id;
  std::string name;
  std::vector<RPCInstance *> children;
  std::vector<RPCInstance *> parents;
  std::vector<RPCInstance *> tmp_children;
  std::vector<RPCInstance *> tmp_parents;
  std::vector<RPCInstance *> tmp_ris;    // pointers to tmp rpc instances
  std::vector<RPCExecution *> tmp_exes;  // pointers to tmp rpc executions

  RPCExecution *rpc_exe_ptr = nullptr;
  RPCExecution *param_sync_rpc_exe_ptr = nullptr;
  bool param_sync = false;
  uint64_t param_sync_size = 0;
  bool offload = false;
  uint64_t offload_size = 0;

  RPCInstance(RPC *rpc_ptr, int id, std::string name);

  uint64_t ready_time = 0, start_time = 0, end_time = 0;

  void remove_parent(RPCInstance *parent);
  void remove_child(RPCInstance *child);
  void add_parent(RPCInstance *parent);
  void add_child(RPCInstance *child);

  void add_tmp_parent(RPCInstance *parent);
  void add_tmp_child(RPCInstance *child);
  void remove_tmp_parent(RPCInstance *parent);
  void remove_tmp_child(RPCInstance *child);

  void resolve_parameter_sync(std::vector<RPCInstance *> tmp_graph,
                              std::unordered_map<std::string, uint64_t> &cost_table);
  // void resolve_offload(std::vector<RPCInstance*> tmp_graph,
  //                      CommStats& comm_stats);
};

uint64_t parameter_sync_cost(uint64_t param_size_bytes, RPCExecution *src, RPCExecution *dst,
                             std::unordered_map<std::string, uint64_t> &cost_table);

uint64_t remote_param_sync_size(uint64_t size, RPCExecution *src, RPCExecution *dst);

// class ModelConfig {
//     std::string model_name;
//     uint64_t param_size_bytes;

//     ModelConfig(std::string model_name, uint64_t param_size_bytes);
// };

#endif