#ifndef SIMULATE_HPP
#define SIMULATE_HPP

#include <rpc.hpp>
#include <device_mesh.hpp>
#include <queue>
#include <iostream>

class SimulateResult {
 public:
  uint64_t end_time;
  bool oom;
  uint64_t mem_cost;
  std::vector<int> index;
  std::vector<RPCExecution *> rpc_exe_list;
  double used_time = 0;

  SimulateResult();

  SimulateResult(uint64_t end_time, bool oom, uint64_t mem_cost, std::vector<int> &index);

  SimulateResult &operator=(const SimulateResult &other);
};

SimulateResult simulate(
    std::vector<RPCInstance *> &graph, std::unordered_map<std::string, uint64_t> &cost_table,
    std::unordered_map<std::string, uint64_t> &model_sizes,
    std::unordered_map<std::string, RPC *> &rpc_table,
    std::unordered_map<std::string, std::vector<RPCExecution *>> &rpc_exe_table,
    std::unordered_map<std::string, std::vector<RPCInstance *>> &ri_table,
    std::unordered_map<std::string, std::vector<RPCInstance *>> &model_name_ri_table,
    std::vector<std::string> &sorted_rpc_names, std::vector<int> &index);

// Comparator for priority queue
struct CompareEndTime {
  bool operator()(SimulateResult const &r1, SimulateResult const &r2) {
    // We want largest end_time at the top of the queue, so we reverse the comparison
    return r1.end_time < r2.end_time;
  }
};

class MinEndTimeQueue {
 public:
  MinEndTimeQueue(int capacity) : k(capacity) {}

  void insert(SimulateResult r) {
    if (queue.size() < k) {
      // std::cout << "push " << "end_time: " << r.end_time << " qsize " << queue.size() <<
      // std::endl;
      queue.push(r);
    } else if (r.end_time < queue.top().end_time) {
      // std::cout << "push " << "end_time: " << r.end_time << " qsize " << queue.size() <<
      // std::endl;
      queue.pop();
      queue.push(r);
    }
  }

  std::priority_queue<SimulateResult, std::vector<SimulateResult>, CompareEndTime> &getQueue() {
    return queue;
  }

 private:
  std::priority_queue<SimulateResult, std::vector<SimulateResult>, CompareEndTime> queue;
  int k;
};

void mergeMinEndTimeQueues(MinEndTimeQueue &target, MinEndTimeQueue &q1);

class CompareReadyTime {
 public:
  bool operator()(RPCInstance *r1, RPCInstance *r2) { return r1->ready_time > r2->ready_time; }
};

#endif  // SIMULATE_HPP