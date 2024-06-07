#include <rpc.hpp>
#include <device_mesh.hpp>
#include <simulate.hpp>
#include <iostream>
#include <queue>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <chrono>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

uint64_t SOFT_GPU_MEM_CAP = 85899345920;  // 80G

SimulateResult::SimulateResult()
    : end_time(std::numeric_limits<uint64_t>::max()), oom(true), mem_cost(0) {}

SimulateResult::SimulateResult(uint64_t end_time, bool oom, uint64_t mem_cost,
                               std::vector<int> &index)
    : end_time(end_time), oom(oom), mem_cost(mem_cost), index(index) {}

SimulateResult simulate(
    std::vector<RPCInstance *> &graph, std::unordered_map<std::string, uint64_t> &cost_table,
    std::unordered_map<std::string, uint64_t> &model_sizes,
    std::unordered_map<std::string, RPC *> &rpc_table,
    std::unordered_map<std::string, std::vector<RPCExecution *>> &rpc_exe_table,
    std::unordered_map<std::string, std::vector<RPCInstance *>> &ri_table,
    std::unordered_map<std::string, std::vector<RPCInstance *>> &model_name_ri_table,
    std::vector<std::string> &sorted_rpc_names, std::vector<int> &index) {
  auto start = std::chrono::high_resolution_clock::now();
  GroupedRPCExecutions grouped_rpc_exe;
  // std::unordered_map<std::string, RPCExecution*> param_dst; // model_name -> rpc_exe_ptr
  std::unordered_set<std::string> offloaded;
  uint64_t oom_penalty = 3;
  int num_rpcs = static_cast<int>(sorted_rpc_names.size());

  for (int i = 0; i < num_rpcs; i++) {
    std::string rpc_name = sorted_rpc_names[i];
    RPC *rpc = rpc_table[rpc_name];
    RPCExecution *rpc_exe = rpc_exe_table[rpc_name][index[i]];

    for (auto &ri : ri_table[rpc_name]) {
      ri->rpc_exe_ptr = rpc_exe;  // assign rpc_exe to rpc_instance
    }

    // dirty implementation, check whether instance is hooked with param syncs
    for (auto &rpc_instance : model_name_ri_table[rpc->model_name]) {
      if (rpc_instance->rpc_ptr->interface_type == "ModelInterfaceType.TRAIN_STEP"
          && rpc->interface_type != "ModelInterfaceType.TRAIN_STEP") {
        // param_dst[rpc -> model_name] = rpc_exe;
        rpc_instance->param_sync = true;
        rpc_instance->param_sync_size = model_sizes[rpc_instance->rpc_ptr->model_name];
        rpc_instance->param_sync_rpc_exe_ptr = rpc_exe;
      }
    }
    grouped_rpc_exe.add(rpc_exe);
    std::string model_name = rpc->model_name;
  }

  // rpc_instances: list of rpc instances, graph
  std::priority_queue<RPCInstance *, std::vector<RPCInstance *>, CompareReadyTime> ready_queue;
  // std::vector<RPCInstance*> executed; // for debug, remove later
  std::unordered_map<std::string, size_t> parent_executed;
  std::unordered_set<DeviceMesh *> device_meshes;

  // for offload and parameter sync RPC instances
  std::vector<RPCInstance *> tmp_graph;

  // resolve parameter sync
  for (RPCInstance *node : graph) {
    tmp_graph.push_back(node);
    node->tmp_children = node->children;
    node->tmp_parents = node->parents;
    // std::cout << "Resolve parameter sync: " << node -> name
    //           << " " << node -> param_sync << std::endl;
    node->resolve_parameter_sync(tmp_graph, cost_table);
    // node -> resolve_offload(tmp_graph, comm_stats);
  }

  uint64_t max_end_time = 0;
  for (RPCInstance *node : tmp_graph) {
    // std::cout << "Node: " << node -> name << " parents: "
    //           << node -> parents.size() << std::endl;
    if (node->parents.size() == 0) ready_queue.push(node);

    // init device meshes
    RPCExecution *rpc_exe = node->rpc_exe_ptr;
    device_meshes.insert(&rpc_exe->device_mesh);
  }

  std::vector<RPCInstance *> executed;

  // simulate
  while (!ready_queue.empty()) {
    RPCInstance *t = ready_queue.top();
    RPCExecution *rpc_exe = t->rpc_exe_ptr;
    uint64_t exec_time = rpc_exe->time_cost;
    DeviceMesh *device_mesh = &rpc_exe->device_mesh;
    ready_queue.pop();

    if (device_mesh->pre_task == nullptr) {
      t->start_time = t->ready_time;
    } else {
      t->start_time = MAX(t->ready_time, device_mesh->pre_task->end_time);
    }
    t->end_time = t->start_time + exec_time;
    max_end_time = MAX(t->end_time, max_end_time);

    for (DeviceMesh *mesh : device_meshes) {
      if (device_mesh->overlap(*mesh)) {
        if (mesh->pre_task == nullptr || mesh->pre_task->end_time <= t->end_time) {
          mesh->pre_task = t;
        }
        // mesh -> pre_task = t;
      }
    }
    executed.push_back(t);

    for (RPCInstance *child : t->tmp_children) {
      child->ready_time = MAX(t->end_time, child->ready_time);
      // std::cout << "parent: " << t -> name
      //           << " child: " << child -> name << std::endl;
      parent_executed[child->name] += 1;
      // child -> remove_parent(t);
      if (child->tmp_parents.size() == parent_executed[child->name]) {
        ready_queue.push(child);
        // std::cout << "Ready: " << child -> name
        //           << " ready time " << child -> ready_time << std::endl;
      }
    }
    // std::cout << "ready_queue size " << ready_queue.size()
    //           << " executed size " << executed.size() << std::endl;
  }

  // 110045999
  // if (max_end_time < 100000000) { // DEBUG
  //     std::cout << "INDEX: [";
  //     for (int i : index) {
  //         std::cout << i << ", ";
  //     }
  //     std::cout << "]" << std::endl;

  //     for (auto& x : rpc_exe_table){
  //         std::string rpc_name = x.first;
  //         std::vector<RPCExecution*>& rpc_exe_list = x.second;
  //         int count = 0;
  //         for (RPCExecution* rpc_exe : rpc_exe_list) {
  //             std::cout << "RPC: " << rpc_name
  //                       << " device mesh " << rpc_exe -> device_mesh.device_mesh_name
  //                       << " time cost " << rpc_exe -> time_cost << std::endl;
  //             count ++;
  //             if (count > 10) break;
  //         }
  //     }

  //     for (RPCInstance* ri : executed) {
  //         for (RPCInstance* parent : ri -> tmp_parents) {
  //             std::cout << "Parent: " << parent -> name << " of " << ri -> name
  //                       << " start time " << parent -> start_time
  //                       << " end time " << parent -> end_time << std::endl;
  //         }

  //         std::cout << "Executed: " << ri -> name
  //                   << " start time " << ri -> start_time
  //                   << " end time " << ri -> end_time
  //                   << " rpc name " << ri -> rpc_ptr -> rpc_name
  //                   << " device mesh "
  //                   << ri -> rpc_exe_ptr -> device_mesh.device_mesh_name
  //                   << " rpc exe time cost " << ri -> rpc_exe_ptr -> time_cost
  //                   << std::endl;
  //     }
  // }

  // clear device mesh pre tasks
  for (DeviceMesh *mesh : device_meshes) { mesh->pre_task = nullptr; }
  // clear rpc instance times
  for (RPCInstance *node : graph) {
    node->tmp_children.clear();
    node->tmp_parents.clear();
    node->ready_time = 0;
    node->start_time = 0;
    node->end_time = 0;
    tmp_graph.clear();

    for (RPCInstance *ptr : node->tmp_ris) { delete ptr; }
    node->tmp_ris.clear();

    for (RPCExecution *ptr : node->tmp_exes) { delete ptr; }
    node->tmp_exes.clear();
  }

  uint64_t current_mem = grouped_rpc_exe.total_mem_cost();
  if (current_mem > SOFT_GPU_MEM_CAP) { max_end_time *= oom_penalty; }
  // std::cout <<     "Max end time: " << max_end_time
  //           << " executed size " << executed.size() << std::endl;
  std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
  // std::cout << "Elapsed time (micro seconds): "
  //           << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()
  //           << std::endl;

  for (OverlapGroup *ptr : grouped_rpc_exe.group.overlap_groups) { delete ptr; }
  grouped_rpc_exe.group.overlap_groups.clear();

  return SimulateResult(max_end_time, current_mem > SOFT_GPU_MEM_CAP, current_mem, index);
};

SimulateResult &SimulateResult::operator=(const SimulateResult &other) {
  if (this != &other) {
    end_time = other.end_time;
    oom = other.oom;
    mem_cost = other.mem_cost;
    index = other.index;
    rpc_exe_list = other.rpc_exe_list;
  }
  return *this;
};

bool isPresent(MinEndTimeQueue &q, SimulateResult element) {
  std::priority_queue<SimulateResult, std::vector<SimulateResult>, CompareEndTime> pq =
      q.getQueue();
  std::queue<SimulateResult> tmp;
  while (!pq.empty()) {
    if (pq.top().end_time == element.end_time) { return true; }
    tmp.push(pq.top());
    pq.pop();
  }
  while (!tmp.empty()) {
    pq.push(tmp.front());
    tmp.pop();
  }
  return false;
}

void mergeMinEndTimeQueues(MinEndTimeQueue &target, MinEndTimeQueue &q1) {
  // Get the underlying priority queues
  std::priority_queue<SimulateResult, std::vector<SimulateResult>, CompareEndTime> pq1 =
      q1.getQueue();

  // Insert all elements from q1 into the merged queue
  while (!pq1.empty()) {
    if (!isPresent(target, pq1.top())) target.insert(pq1.top());
    pq1.pop();
  }
}