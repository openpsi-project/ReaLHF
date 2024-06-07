#include <iostream>
#include <iomanip>
#include <algorithm>
#include <rpc.hpp>
#include <device_mesh.hpp>
#include <simulate.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <chrono>
#include <limits>
#include <fstream>
#include <cmath>
#include <random>
#include <functional>
#include <thread>

namespace py = pybind11;

uint64_t VALID_COUNT_CAP = 25000000;  // 25000000
size_t MAX_EXE_PER_RPC = 1000;
// std::unordered_map<std::string, DeviceMesh*> device_mesh_map;

void print_int_vector(std::vector<int> &vec) {
  std::cout << "[";
  for (int i = 0; i < static_cast<int>(vec.size()); i++) { std::cout << vec[i] << ", "; }
  std::cout << "] ";
};

std::size_t vector_hash(const std::vector<int> &vec) {
  std::size_t seed = vec.size();
  for (const auto &i : vec) {
    seed ^= std::hash<int>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}

long check_memory_bytes() {
  std::ifstream statm_file("/proc/self/statm");
  long memory_usage_bytes = -1;
  if (!statm_file) {
    std::cerr << "Failed to open /proc/self/statm\n";
  } else {
    long size, resident, share, text, lib, data, dt;
    statm_file >> size >> resident >> share >> text >> lib >> data >> dt;

    // size is in pages, to convert to bytes, multiply by the page size
    long page_size = sysconf(_SC_PAGESIZE);
    memory_usage_bytes = size * page_size;
  }
  return memory_usage_bytes;
};

void make_rpc_exe_table(std::unordered_map<std::string, std::vector<RPCExecution *>> &rpc_exe_table,
                        std::vector<RPCExecution *> &rpc_exes) {
  for (auto &rpc_exe : rpc_exes) {
    if (rpc_exe_table[rpc_exe->rpc_ptr->rpc_name].size() < MAX_EXE_PER_RPC)
      rpc_exe_table[rpc_exe->rpc_ptr->rpc_name].push_back(rpc_exe);
  }

  for (auto &x : rpc_exe_table) {
    std::string rpc_name = x.first;
    std::vector<RPCExecution *> &rpc_exe_list = x.second;
    // sort first
    std::sort(rpc_exe_list.begin(), rpc_exe_list.end(),
              [](const RPCExecution *a, const RPCExecution *b) {
                if (a->time_cost == b->time_cost)
                  return a->device_mesh.name < b->device_mesh.name;
                else
                  return a->time_cost < b->time_cost;
              });
  }
}

void make_sorted_rpc_names(
    std::vector<std::string> &sorted_rpc_names,
    std::unordered_map<std::string, std::vector<RPCExecution *>> &rpc_exe_table) {
  std::vector<std::pair<std::string, uint64_t>> average_time_cost;
  for (auto &x : rpc_exe_table) {
    std::string rpc_name = x.first;
    std::vector<RPCExecution *> &rpc_exe_list = x.second;
    uint64_t total_time_cost = 0;
    int c = 0;
    for (auto &rpc_exe : rpc_exe_list) {
      total_time_cost += rpc_exe->time_cost;
      c += 1;
      if (c > 10) break;
    }
    average_time_cost.push_back(std::make_pair(rpc_name, total_time_cost / 10));
  }

  std::sort(average_time_cost.begin(), average_time_cost.end(),
            [](const std::pair<std::string, float> &a, const std::pair<std::string, float> &b) {
              return a.second > b.second;
            });

  for (auto &x : average_time_cost) { sorted_rpc_names.push_back(x.first); }
}

void prepare(std::unordered_map<std::string, std::vector<RPCExecution *>> &rpc_exe_table,
             std::unordered_map<std::string, RPC *> &rpc_table,
             std::vector<std::string> &sorted_rpc_names,
             std::unordered_map<std::string, std::vector<RPCInstance *>> &ri_table,
             std::unordered_map<std::string, std::vector<RPCInstance *>> &model_name_ri_table,
             std::vector<RPC *> rpcs, std::vector<RPCExecution *> rpc_exes,
             std::vector<RPCInstance *> graph) {
  std::vector<std::pair<std::string, uint64_t>> average_time_cost;
  for (auto &rpc : rpcs) { rpc_table[rpc->rpc_name] = rpc; }

  make_rpc_exe_table(rpc_exe_table, rpc_exes);
  make_sorted_rpc_names(sorted_rpc_names, rpc_exe_table);

  for (auto &rpc_instance : graph) {
    ri_table[rpc_instance->rpc_ptr->rpc_name].push_back(rpc_instance);
  }

  for (auto &rpc_instance : graph) {
    model_name_ri_table[rpc_instance->rpc_ptr->model_name].push_back(rpc_instance);
  }
};

std::vector<SimulateResult> mcmc_search(std::vector<RPC *> rpcs,
                                        std::vector<RPCExecution *> rpc_exes,
                                        std::vector<RPCInstance *> graph,
                                        std::unordered_map<std::string, uint64_t> &cost_table,
                                        std::unordered_map<std::string, uint64_t> model_sizes,
                                        double beta, double time_limit,
                                        MinEndTimeQueue &top_k_queue) {
  std::unordered_map<std::string, std::vector<RPCExecution *>> rpc_exe_table;
  std::unordered_map<std::string, RPC *> rpc_table;
  std::vector<std::string> sorted_rpc_names;
  std::unordered_map<std::string, std::vector<RPCInstance *>> ri_table;
  std::unordered_map<std::string, std::vector<RPCInstance *>> model_name_ri_table;
  std::chrono::duration<double> time_limit_duration(time_limit);
  std::vector<SimulateResult> time_cost_cache;

  prepare(rpc_exe_table, rpc_table, sorted_rpc_names, ri_table, model_name_ri_table, rpcs, rpc_exes,
          graph);

  std::vector<int> index;
  std::vector<int> min_index;
  std::vector<int> max_index;
  uint64_t min_index_mem = 0;
  uint64_t valid_count = 0;
  uint64_t oom_count = 0;
  int num_rpcs = static_cast<int>(sorted_rpc_names.size());
  uint64_t min_time_cost = std::numeric_limits<uint64_t>::max();
  uint64_t max_time_cost = 0;
  double avg = 0;

  // [6, 2, 3, 2, 12, 15, ]
  // [0, 0, 0, 7, 8, 13, ]
  // index = { 23, 7, 30, 154, 173, 265 };
  // SimulateResult sr1 = simulate(graph, cost_table, model_sizes,
  //                               rpc_table, rpc_exe_table, ri_table,
  //                               model_name_ri_table, sorted_rpc_names,
  //                               index);
  // std::cout << "index 1 end time " << sr1.end_time << " mem cost " << sr1.mem_cost << std::endl;
  // std::cout << "************************" << std::endl;
  // index = { 1, 0, 0, 11, 20, 3   };
  // SimulateResult sr2 = simulate(graph, cost_table, model_sizes,
  //                               rpc_table, rpc_exe_table, ri_table,
  //                               model_name_ri_table, sorted_rpc_names,
  //                               index);
  // std::cout << "index 2 end time " << sr2.end_time << " mem cost " << sr2.mem_cost << std::endl;
  // exit(0);

  // initial value
  index.resize(sorted_rpc_names.size(), 0);

  // index 1
  SimulateResult first_sr = simulate(graph, cost_table, model_sizes, rpc_table, rpc_exe_table,
                                     ri_table, model_name_ri_table, sorted_rpc_names, index);
  SimulateResult final_sr = first_sr;
  uint64_t current_cost = first_sr.end_time;
  time_cost_cache.push_back(first_sr);

  if (first_sr.oom)
    oom_count += 1;
  else
    top_k_queue.insert(first_sr);
  // std::cout << "initial cost " << current_cost << " oom "
  //           << first_sr.oom << std::endl;
  // index = {0, 1, 3, 34, 4, 10};
  // std::unordered_map<std::size_t, uint64_t> time_cost_cache;

  auto start = std::chrono::high_resolution_clock::now();
  // bool outer_loop_break_flag = false;
  while (valid_count < VALID_COUNT_CAP) {
    // only change one model execution in each iteration
    // std::vector<SimulateResult> sr_vector;
    std::unordered_map<int, std::pair<int, int>> flatten_to_pair;
    std::vector<double> weight;
    // double beta = 0.0075;
    int max_step_range = 10000;
    int current = 0;

    std::vector<int> new_index(index);
    for (int i = 0; i < num_rpcs; i++) {
      std::string rpc_name = sorted_rpc_names[i];
      int c_i = index[i];
      int min_i = std::max(0, c_i - max_step_range);
      int max_i =
          std::min(static_cast<int>(rpc_exe_table[rpc_name].size()), c_i + max_step_range + 1);

      for (int j = min_i; j < max_i; j++) {
        if (j == c_i) continue;
        // int tmp = new_index[i];
        // new_index[i] = j;
        // SimulateResult sr = simulate(graph, cost_table, model_sizes,
        //                              rpc_table, rpc_exe_table, ri_table,
        //                              model_name_ri_table, sorted_rpc_names,
        //                              new_index);
        // sr_vector.push_back(sr);
        // new_index[i] = tmp;
        flatten_to_pair[current] = std::make_pair(i, j);
        current++;
      }
    }

    // if (time_cost_cache.size() > 10000000) {
    //     time_cost_cache.clear();
    // }

    // std::cout << "sr vector size " << sr_vector.size() << std::endl;
    // for (int i = 0; i < static_cast<int>(sr_vector.size()); i++) {
    //     weight.push_back(std::exp(-beta * (sr_vector[i].end_time/100000)));
    // }
    // exit(0);

    std::random_device rd;
    std::mt19937 gen(rd());
    // std::discrete_distribution<int> d(weight.begin(), weight.end());
    std::uniform_int_distribution<int> d(0, static_cast<int>(flatten_to_pair.size() - 1));
    int selected = d(gen);

    // assign new index
    int selected_i = flatten_to_pair[selected].first;
    int selected_j = flatten_to_pair[selected].second;
    new_index[selected_i] = selected_j;

    SimulateResult selected_sr =
        simulate(graph, cost_table, model_sizes, rpc_table, rpc_exe_table, ri_table,
                 model_name_ri_table, sorted_rpc_names, new_index);
    uint64_t selected_cost = selected_sr.end_time;
    // if (selected_sr.oom) {
    //     std::cout << "oom max end time " << selected_cost << std::endl;
    // } else {
    //     std::cout << "max end time " << selected_cost << std::endl;
    // }

    if (selected_cost < std::numeric_limits<uint64_t>::max()) {
      bool accepted = true;
      if (current_cost < selected_cost) {
        double accept_prob = std::exp(-beta * ((selected_cost - current_cost) / 100000));
        // double accept_prob = ap * ap;
        std::bernoulli_distribution accept_dist(accept_prob);
        accepted = accept_dist(gen);
      }

      if (accepted) {
        // std::cout << "accepted" << std::endl;
        index = new_index;
        current_cost = selected_cost;
        valid_count++;
        if (!selected_sr.oom) {
          // min_time_cost = std::min(min_time_cost, selected_cost);
          // max_time_cost = std::max(max_time_cost, selected_cost);
          avg = (selected_cost + avg * (valid_count - 1)) / valid_count;
          if (min_time_cost > selected_cost) {
            min_time_cost = selected_cost;
            min_index = index;
            min_index_mem = selected_sr.mem_cost;
            // final_sr = selected_sr;
            auto now = std::chrono::high_resolution_clock::now();
            double diff =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            selected_sr.used_time = diff;
            time_cost_cache.push_back(selected_sr);
          }
          if (min_time_cost == selected_cost) {
            if (min_index_mem > selected_sr.mem_cost) {
              min_index = index;
              min_index_mem = selected_sr.mem_cost;
              // final_sr = selected_sr;
            }
          }
          top_k_queue.insert(selected_sr);
          if (max_time_cost < selected_cost) {
            max_time_cost = selected_cost;
            max_index = index;
          }
        } else {
          oom_count += 1;
        }
        // if (min_time_cost <= 100000000) break; // DEBUG

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        if (valid_count % 1000 == 0) {
          std::cout << " valid_count " << valid_count << " oom count " << oom_count << " time "
                    << diff.count() << " min time cost " << min_time_cost << " min index mem cost "
                    << min_index_mem << " max time cost " << max_time_cost << " current cost "
                    << current_cost << " avg " << avg << " mem usage " << check_memory_bytes()
                    << std::endl;

          std::cout << "min index : ";
          print_int_vector(min_index);
          std::cout << std::endl;
          std::cout << "max index : ";
          print_int_vector(max_index);
          std::cout << std::endl;
          std::cout << "current index : ";
          print_int_vector(index);
          std::cout << std::endl;
        }
        if (diff > time_limit_duration) break;
      }
    }
  }

  // std::cout << "break" << std::endl;

  // int rpc_index = 0;
  // for (int index : final_sr.index) {
  //     // std::cout << "index " << index << " rpc_index " << rpc_index << std::endl;
  //     // std::cout << "rpc name " << sorted_rpc_names[rpc_index] << std::endl;
  //     RPCExecution* re_ptr = rpc_exe_table[sorted_rpc_names[rpc_index]][index];
  //     final_sr.rpc_exe_list.push_back(re_ptr);
  //     rpc_index ++;
  // }
  // std::cout << "final_sr.rpc_exe_list size " << final_sr.rpc_exe_list.size() << std::endl;

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cout << "MCMC Search finished, beta " << beta << " time limit " << time_limit << " seconds"
            << " valid_count " << valid_count << " oom_count " << oom_count << " time "
            << diff.count() << " min time cost " << min_time_cost << " max time cost "
            << max_time_cost << " avg " << avg << std::endl;
  std::cout << "RPCExecutions: " << std::endl;
  for (auto &re_ptr : final_sr.rpc_exe_list) { std::cout << re_ptr->to_string() << std::endl; }
  return time_cost_cache;
  // return final_sr;
}

void multi_mcmc_search(std::vector<RPC *> rpcs, std::vector<RPCExecution *> rpc_exes,
                       std::vector<RPCInstance *> graph,
                       std::unordered_map<std::string, uint64_t> &cost_table,
                       std::unordered_map<std::string, uint64_t> model_sizes, double beta_min,
                       double beta_max, double beta_step, MinEndTimeQueue &res_queue,
                       int top_k = 10, double time_limit = 60.0,
                       int repeat = 1  // Remove the trailing comma here
) {
  SimulateResult sr;
  std::vector<MinEndTimeQueue *> queues;
  // std::vector<std::thread> ts;
  for (int i = 0; i < repeat; i++) {
    for (double beta = beta_min; beta < beta_max; beta += beta_step) {
      MinEndTimeQueue *q = new MinEndTimeQueue(10);
      queues.push_back(q);

      // Create a new thread to run mcmc_search
      std::vector<SimulateResult> r =
          mcmc_search(rpcs, rpc_exes, graph, cost_table, model_sizes, beta, time_limit, *q);
    }
  }

  for (auto &q : queues) {
    mergeMinEndTimeQueues(res_queue, *q);
    // delete q;
  }

  // std::cout << "Best result: " << sr.end_time << std::endl;
  // for (auto& re_ptr : sr.rpc_exe_list) {
  //     std::cout << re_ptr -> to_string() << std::endl;
  // }
  // std::cout << "Index: ";
  // for (int i : sr.index) {
  //     std::cout << i << " ";
  // }
  // std::cout << std::endl;
  // std::cout << "Time cost: " << sr.end_time
  //           << ", mem cost: " << sr.mem_cost << std::endl;
  // std::cout << "sr.rpc_exe_list size " << sr.rpc_exe_list.size() << std::endl;
  // return sr;
}

void input_check(std::vector<RPC *> rpcs, std::vector<RPCExecution *> rpc_exes,
                 std::vector<RPCInstance *> graph, CommStats &comm_stats) {
  std::cout << "rpcs: " << rpcs.size() << std::endl;
  std::cout << "rpc_exes: " << rpc_exes.size() << std::endl;
  std::cout << "graph: " << graph.size() << std::endl;

  for (auto &rpc_instance : graph) {
    std::cout << "==================" << std::endl;
    std::cout << "rpc instance: " << rpc_instance->name << std::endl;
    std::cout << "parents" << std::endl;
    for (auto &parent : rpc_instance->parents) { std::cout << parent->name << " "; }
    std::cout << std::endl;
    std::cout << "children" << std::endl;
    for (auto &child : rpc_instance->children) { std::cout << child->name << " "; }
    std::cout << std::endl;
    // std::cout << "parents: " << rpc_instance -> parents.size()
    //           << " children: " << rpc_instance -> children.size() << std::endl;
  }

  std::cout << "comm_stats: " << std::endl;
  std::cout << "local_send: " << comm_stats.local_send << std::endl;
  std::cout << "local_recv: " << comm_stats.local_recv << std::endl;
  std::cout << "remote_send: " << comm_stats.remote_send << std::endl;
  std::cout << "remote_recv: " << comm_stats.remote_recv << std::endl;
  std::cout << "offload_store: " << comm_stats.offload_store << std::endl;
  std::cout << "offload_load: " << comm_stats.offload_load << std::endl;
}

RPC *cast_rpc(py::handle rpc_py) {
  return new RPC(py::str(rpc_py.attr("model_name")).cast<std::string>(),
                 rpc_py.attr("name").cast<std::string>(),
                 py::str(rpc_py.attr("interface_type")).cast<std::string>());
}

DeviceMesh *cast_device_mesh(py::handle device_mesh_py,
                             std::unordered_map<std::string, DeviceMesh *> &device_mesh_map) {
  std::string name = device_mesh_py.attr("name").cast<std::string>();
  if (device_mesh_map.find(name) == device_mesh_map.end()) {
    py::array_t<int32_t> mapping_array =
        device_mesh_py.attr("mapping").cast<pybind11::array_t<int32_t>>();
    py::buffer_info buf_info = mapping_array.request();

    auto rows = buf_info.shape[0];
    auto cols = buf_info.shape[1];

    std::vector<std::vector<int>> mapping(rows, std::vector<int>(cols));

    // Get a pointer to the data
    int32_t *data = static_cast<int32_t *>(buf_info.ptr);

    // Fill the 2D vector with data from the numpy array
    for (size_t i = 0; i < static_cast<size_t>(rows); ++i) {
      for (size_t j = 0; j < static_cast<size_t>(cols); ++j) { mapping[i][j] = data[i * cols + j]; }
    }

    DeviceMesh *device_mesh =
        new DeviceMesh(device_mesh_py.attr("n_nodes").cast<int>(),
                       device_mesh_py.attr("n_gpus_per_node").cast<int>(), mapping,
                       device_mesh_py.attr("global_mesh_name").cast<std::string>(),
                       device_mesh_py.attr("name").cast<std::string>());

    device_mesh_map[name] = device_mesh;
    return device_mesh;
  } else {
    return device_mesh_map[name];
  }
}

ModelParallelStrategy *cast_model_parallel_strategy(py::handle model_parallel_strategy_py) {
  return new ModelParallelStrategy(
      model_parallel_strategy_py.attr("pipeline_parallel_size").cast<int>(),
      model_parallel_strategy_py.attr("data_parallel_size").cast<int>(),
      model_parallel_strategy_py.attr("model_parallel_size").cast<int>());
}

RPCExecution *cast_rpc_execution(py::handle rpc_exe_py, std::unordered_map<std::string, RPC *> &tmp,
                                 std::unordered_map<std::string, DeviceMesh *> &device_mesh_map) {
  DeviceMesh *device_mesh = cast_device_mesh(rpc_exe_py.attr("device_mesh"), device_mesh_map);
  ModelParallelStrategy *model_parallel_strategy =
      cast_model_parallel_strategy(rpc_exe_py.attr("parallel_strategy"));
  // RPC* rpc = cast_rpc(rpc_exe_py.attr("rpc"));

  return new RPCExecution(
      tmp[rpc_exe_py.attr("rpc").attr("name").cast<std::string>()], *device_mesh,
      *model_parallel_strategy, rpc_exe_py.attr("time_cost").cast<uint64_t>(),
      rpc_exe_py.attr("mem").cast<uint64_t>(), rpc_exe_py.attr("static_mem").cast<uint64_t>());
}

RPCInstance *cast_rpc_instance_wo_dependency(py::handle rpc_instance_py,
                                             std::unordered_map<std::string, RPC *> &tmp) {
  return new RPCInstance(tmp[rpc_instance_py.attr("rpc").attr("name").cast<std::string>()],
                         rpc_instance_py.attr("iteration_id").cast<int>(),
                         rpc_instance_py.attr("name").cast<std::string>());
}

void cast_rpc_instance_dependency(py::handle rpc_instance_py, RPCInstance *ri_ptr,
                                  std::unordered_map<std::string, RPCInstance *> &tmp_graph) {
  for (py::handle parent_py : rpc_instance_py.attr("parents"))
    ri_ptr->parents.push_back(tmp_graph[parent_py.attr("name").cast<std::string>()]);
  for (py::handle child_py : rpc_instance_py.attr("children"))
    ri_ptr->children.push_back(tmp_graph[child_py.attr("name").cast<std::string>()]);
}

py::list py_single_mcmc_search_time_profile(py::list rpcs_py, py::list rpc_exes_py,
                                            py::list graph_py, py::dict cost_table_py,
                                            py::dict model_sizes_py, py::object beta,
                                            py::object time_limit) {
  std::vector<RPC *> rpcs;
  std::unordered_map<std::string, RPC *> tmp;
  for (py::handle rpc_py : rpcs_py) {
    RPC *rpc_ptr = cast_rpc(rpc_py);
    rpcs.push_back(rpc_ptr);
    tmp[rpc_ptr->rpc_name] = rpc_ptr;
  }

  std::vector<RPCExecution *> rpc_exes;
  std::unordered_map<std::string, DeviceMesh *> tmp_device_mesh;
  for (py::handle rpc_exe_py : rpc_exes_py) {
    RPCExecution *rpc_exe_ptr = cast_rpc_execution(rpc_exe_py, tmp, tmp_device_mesh);
    rpc_exes.push_back(rpc_exe_ptr);
  }

  std::vector<RPCInstance *> graph;
  std::unordered_map<std::string, RPCInstance *> tmp_graph;
  for (py::handle ri_py : graph_py) {
    RPCInstance *ri_ptr = cast_rpc_instance_wo_dependency(ri_py, tmp);
    // std::cout << "cast " << ri_ptr -> name << std::endl;
    tmp_graph[ri_ptr->name] = ri_ptr;
  }
  // build dependecny
  for (py::handle ri_py : graph_py) {
    std::string ri_name = ri_py.attr("name").cast<std::string>();
    cast_rpc_instance_dependency(ri_py, tmp_graph[ri_name], tmp_graph);
    graph.push_back(tmp_graph[ri_name]);
  }

  std::unordered_map<std::string, uint64_t> cost_table =
      cost_table_py.cast<std::unordered_map<std::string, uint64_t>>();

  std::unordered_map<std::string, uint64_t> model_sizes =
      model_sizes_py.cast<std::unordered_map<std::string, uint64_t>>();
  MinEndTimeQueue res_queue(10);
  std::vector<SimulateResult> rlist =
      mcmc_search(rpcs, rpc_exes, graph, cost_table, model_sizes, beta.cast<double>(),
                  time_limit.cast<double>(), res_queue);

  std::unordered_map<std::string, std::vector<RPCExecution *>> rpc_exe_table;
  std::vector<std::string> sorted_rpc_names;
  make_rpc_exe_table(rpc_exe_table, rpc_exes);
  make_sorted_rpc_names(sorted_rpc_names, rpc_exe_table);
  py::list result;
  std::cout << "rlist.size " << rlist.size() << std::endl;
  for (auto &r : rlist) {
    // SimulateResult r = res_queue.getQueue().top();
    // res_queue.getQueue().pop();

    std::cout << "End time: " << r.end_time << std::endl;
    for (auto &re_ptr : r.rpc_exe_list) { std::cout << re_ptr->to_string() << std::endl; }
    std::cout << "Index: ";
    for (int i : r.index) { std::cout << i << " "; }
    std::cout << std::endl;
    std::cout << "Time cost: " << r.end_time << ", mem cost: " << r.mem_cost << std::endl;

    int rpc_index = 0;
    for (int index : r.index) {
      // std::cout << "index " << index << " rpc_index " << rpc_index << std::endl;
      // std::cout << "rpc name " << sorted_rpc_names[rpc_index] << std::endl;
      RPCExecution *re_ptr = rpc_exe_table[sorted_rpc_names[rpc_index]][index];
      r.rpc_exe_list.push_back(re_ptr);
      rpc_index++;
    }

    py::dict rdict;
    for (auto &re_ptr : r.rpc_exe_list) {
      py::dict rpc_exe_info;
      std::string rpc_name = re_ptr->rpc_ptr->rpc_name;
      py::object rpc_name_obj = py::str(rpc_name);
      // rpc_exe_info.append(re_ptr -> device_mesh.device_mesh_name);
      // rpc_exe_info.append(re_ptr -> model_parallel_strategy.num_dp);
      // rpc_exe_info.append(re_ptr -> model_parallel_strategy.num_mp);
      // rpc_exe_info.append(re_ptr -> model_parallel_strategy.num_pp);
      rpc_exe_info["device_mesh"] = re_ptr->device_mesh.name;
      rpc_exe_info["num_dp"] = re_ptr->model_parallel_strategy.num_dp;
      rpc_exe_info["num_mp"] = re_ptr->model_parallel_strategy.num_mp;
      rpc_exe_info["num_pp"] = re_ptr->model_parallel_strategy.num_pp;
      rdict[rpc_name_obj] = rpc_exe_info;
      // std::cout << "append key " << rpc_name_obj << std::endl;
    }
    rdict["end_time"] = r.end_time;
    rdict["mem_cost"] = r.mem_cost;
    rdict["used_time"] = r.used_time;
    result.append(rdict);
  }
  return result;
};

py::list py_multi_mcmc_search(py::list rpcs_py, py::list rpc_exes_py, py::list graph_py,
                              py::dict cost_table_py, py::dict model_sizes_py,
                              py::object beta_min_py, py::object beta_max_py,
                              py::object beta_step_py, py::object time_limit_py,
                              py::object repeat) {
  std::vector<RPC *> rpcs;
  std::unordered_map<std::string, RPC *> tmp;
  for (py::handle rpc_py : rpcs_py) {
    RPC *rpc_ptr = cast_rpc(rpc_py);
    rpcs.push_back(rpc_ptr);
    tmp[rpc_ptr->rpc_name] = rpc_ptr;
  }

  std::vector<RPCExecution *> rpc_exes;
  std::unordered_map<std::string, DeviceMesh *> tmp_device_mesh;
  for (py::handle rpc_exe_py : rpc_exes_py) {
    RPCExecution *rpc_exe_ptr = cast_rpc_execution(rpc_exe_py, tmp, tmp_device_mesh);
    rpc_exes.push_back(rpc_exe_ptr);
  }

  std::vector<RPCInstance *> graph;
  std::unordered_map<std::string, RPCInstance *> tmp_graph;
  for (py::handle ri_py : graph_py) {
    RPCInstance *ri_ptr = cast_rpc_instance_wo_dependency(ri_py, tmp);
    // std::cout << "cast " << ri_ptr -> name << std::endl;
    tmp_graph[ri_ptr->name] = ri_ptr;
  }
  // build dependecny
  for (py::handle ri_py : graph_py) {
    std::string ri_name = ri_py.attr("name").cast<std::string>();
    cast_rpc_instance_dependency(ri_py, tmp_graph[ri_name], tmp_graph);
    graph.push_back(tmp_graph[ri_name]);
  }

  std::unordered_map<std::string, uint64_t> cost_table =
      cost_table_py.cast<std::unordered_map<std::string, uint64_t>>();

  std::unordered_map<std::string, uint64_t> model_sizes =
      model_sizes_py.cast<std::unordered_map<std::string, uint64_t>>();

  double beta_min = beta_min_py.cast<double>();
  double beta_max = beta_max_py.cast<double>();
  double beta_step = beta_step_py.cast<double>();
  double time_limit = time_limit_py.cast<double>();
  int rp = repeat.cast<int>();

  MinEndTimeQueue res_queue(10);
  multi_mcmc_search(rpcs, rpc_exes, graph, cost_table, model_sizes, beta_min, beta_max, beta_step,
                    res_queue, 10, time_limit, rp);

  std::unordered_map<std::string, std::vector<RPCExecution *>> rpc_exe_table;
  std::vector<std::string> sorted_rpc_names;
  make_rpc_exe_table(rpc_exe_table, rpc_exes);
  make_sorted_rpc_names(sorted_rpc_names, rpc_exe_table);

  // std::cout << "r.rpc_exe_list size " << r.rpc_exe_list.size() << std::endl;
  // for (int rpc_index = 0; rpc_index < 6; rpc_index++) {
  //     std::cout << "rpc name " << sorted_rpc_names[rpc_index] << std::endl;
  //     int index = 0;
  //     for (auto& re_ptr : rpc_exe_table[sorted_rpc_names[rpc_index]]) {
  //         std::cout << "index " << index << " " << re_ptr -> to_string() << std::endl;
  //         index ++;
  //     }
  // }

  py::list result;
  std::cout << "res_queue.getQueue().size " << res_queue.getQueue().size() << std::endl;
  while (!res_queue.getQueue().empty()) {
    SimulateResult r = res_queue.getQueue().top();
    res_queue.getQueue().pop();

    std::cout << "End time: " << r.end_time << std::endl;
    for (auto &re_ptr : r.rpc_exe_list) { std::cout << re_ptr->to_string() << std::endl; }
    std::cout << "Index: ";
    for (int i : r.index) { std::cout << i << " "; }
    std::cout << std::endl;
    std::cout << "Time cost: " << r.end_time << ", mem cost: " << r.mem_cost << std::endl;

    int rpc_index = 0;
    for (int index : r.index) {
      // std::cout << "index " << index << " rpc_index " << rpc_index << std::endl;
      // std::cout << "rpc name " << sorted_rpc_names[rpc_index] << std::endl;
      RPCExecution *re_ptr = rpc_exe_table[sorted_rpc_names[rpc_index]][index];
      r.rpc_exe_list.push_back(re_ptr);
      rpc_index++;
    }

    py::dict rdict;
    for (auto &re_ptr : r.rpc_exe_list) {
      py::dict rpc_exe_info;
      std::string rpc_name = re_ptr->rpc_ptr->rpc_name;
      py::object rpc_name_obj = py::str(rpc_name);
      // convert device mesh mapping into py::array_t<int>
      std::vector<std::vector<int>> mapping = re_ptr->device_mesh.mapping;
      int rows = mapping.size();
      int cols = mapping[0].size();

      py::array_t<int> numpy_array({rows, cols});
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          *numpy_array.mutable_data(i, j) = mapping[i][j];
          // std::cout << i << j << mapping[i][j] << std::endl;
        }
      }
      // store in py::dict
      rpc_exe_info["device_mesh_mapping"] = numpy_array;
      rpc_exe_info["device_mesh_name"] = re_ptr->device_mesh.name;
      rpc_exe_info["num_dp"] = re_ptr->model_parallel_strategy.num_dp;
      rpc_exe_info["num_mp"] = re_ptr->model_parallel_strategy.num_mp;
      rpc_exe_info["num_pp"] = re_ptr->model_parallel_strategy.num_pp;
      rdict[rpc_name_obj] = rpc_exe_info;
      // std::cout << "append key " << rpc_name_obj << std::endl;
    }
    rdict["end_time"] = r.end_time;
    rdict["mem_cost"] = r.mem_cost;
    result.append(rdict);
  }

  return result;
};

PYBIND11_MODULE(mdm_search, m) {
  m.doc() = "model device mapping search module";

  // for debug
  // m.def("mcmc_search", [](py::list rpcs_py, py::list rpc_exes_py,
  //                         py::list graph_py, py::object comm_stats_py,
  //                         py::dict model_sizes_py, py::object beta_py,
  //                         py::object time_limit_py) {
  //     std::vector<RPC*> rpcs;
  //     std::unordered_map<std::string, RPC*> tmp;
  //     for (py::handle rpc_py : rpcs_py) {
  //         RPC* rpc_ptr = cast_rpc(rpc_py);
  //         rpcs.push_back(rpc_ptr);
  //         tmp[rpc_ptr -> rpc_name] = rpc_ptr;
  //     }

  //     std::vector<RPCExecution*> rpc_exes;
  //     for (py::handle rpc_exe_py : rpc_exes_py) {
  //         RPCExecution* rpc_exe_ptr = cast_rpc_execution(rpc_exe_py, tmp);
  //         rpc_exes.push_back(rpc_exe_ptr);
  //     }

  //     std::vector<RPCInstance*> graph;
  //     std::unordered_map<std::string, RPCInstance*> tmp_graph;
  //     for (py::handle ri_py : graph_py) {
  //         RPCInstance* ri_ptr = cast_rpc_instance_wo_dependency(ri_py, tmp);
  //         std::cout << "cast " << ri_ptr -> name << std::endl;
  //         tmp_graph[ri_ptr -> name] = ri_ptr;
  //     }
  //     // build dependecny
  //     for (py::handle ri_py : graph_py) {
  //         std::string ri_name = ri_py.attr("name").cast<std::string>();
  //         cast_rpc_instance_dependency(ri_py, tmp_graph[ri_name], tmp_graph);
  //         graph.push_back(tmp_graph[ri_name]);
  //     }

  //     CommStats comm_stats(
  //         comm_stats_py.attr("local_send").cast<uint64_t>(),
  //         comm_stats_py.attr("local_recv").cast<uint64_t>(),
  //         comm_stats_py.attr("remote_send").cast<uint64_t>(),
  //         comm_stats_py.attr("remote_recv").cast<uint64_t>(),
  //         comm_stats_py.attr("offload_load").cast<uint64_t>(),
  //         comm_stats_py.attr("offload_store").cast<uint64_t>()
  //     );

  //     std::unordered_map<std::string, uint64_t> model_sizes
  //         = model_sizes_py.cast<std::unordered_map<std::string, uint64_t>>();

  //     double beta = beta_py.cast<double>();
  //     double time_limit = time_limit_py.cast<double>();

  //     mcmc_search(rpcs, rpc_exes, graph,
  //                 comm_stats, model_sizes, beta,
  //                 time_limit);
  // });

  m.def("input_check",
        [](py::list rpcs_py, py::list rpc_exes_py, py::list graph_py, py::object comm_stats_py) {
          std::vector<RPC *> rpcs;
          std::unordered_map<std::string, RPC *> tmp;
          for (py::handle rpc_py : rpcs_py) {
            RPC *rpc_ptr = cast_rpc(rpc_py);
            rpcs.push_back(rpc_ptr);
            tmp[rpc_ptr->rpc_name] = rpc_ptr;
          }

          std::vector<RPCExecution *> rpc_exes;
          std::unordered_map<std::string, DeviceMesh *> tmp_device_mesh;
          for (py::handle rpc_exe_py : rpc_exes_py) {
            RPCExecution *rpc_exe_ptr = cast_rpc_execution(rpc_exe_py, tmp, tmp_device_mesh);
            rpc_exes.push_back(rpc_exe_ptr);
          }

          std::vector<RPCInstance *> graph;
          std::unordered_map<std::string, RPCInstance *> tmp_graph;
          for (py::handle ri_py : graph_py) {
            RPCInstance *ri_ptr = cast_rpc_instance_wo_dependency(ri_py, tmp);
            std::cout << "cast " << ri_ptr->name << std::endl;
            tmp_graph[ri_ptr->name] = ri_ptr;
          }
          // build dependecny
          for (py::handle ri_py : graph_py) {
            std::string ri_name = ri_py.attr("name").cast<std::string>();
            cast_rpc_instance_dependency(ri_py, tmp_graph[ri_name], tmp_graph);
            graph.push_back(tmp_graph[ri_name]);
          }

          CommStats comm_stats(comm_stats_py.attr("local_send").cast<uint64_t>(),
                               comm_stats_py.attr("local_recv").cast<uint64_t>(),
                               comm_stats_py.attr("remote_send").cast<uint64_t>(),
                               comm_stats_py.attr("remote_recv").cast<uint64_t>(),
                               comm_stats_py.attr("offload_load").cast<uint64_t>(),
                               comm_stats_py.attr("offload_store").cast<uint64_t>());

          input_check(rpcs, rpc_exes, graph, comm_stats);
        });

  // mcmc search to py result
  m.def("multi_mcmc_search", &py_multi_mcmc_search);

  m.def("parameter_sync_cost", [](py::object rpcs_py, py::object param_size_bytes_py,
                                  py::dict cost_table_py, py::object src_py, py::object dst_py) {
    uint64_t param_size_bytes = param_size_bytes_py.cast<uint64_t>();
    std::unordered_map<std::string, uint64_t> cost_table =
        cost_table_py.cast<std::unordered_map<std::string, uint64_t>>();
    std::vector<RPC *> rpcs;
    std::unordered_map<std::string, RPC *> tmp;
    for (py::handle rpc_py : rpcs_py) {
      RPC *rpc_ptr = cast_rpc(rpc_py);
      rpcs.push_back(rpc_ptr);
      tmp[rpc_ptr->rpc_name] = rpc_ptr;
    }

    std::unordered_map<std::string, DeviceMesh *> tmp_device_mesh;
    RPCExecution *src = cast_rpc_execution(src_py, tmp, tmp_device_mesh);
    RPCExecution *dst = cast_rpc_execution(dst_py, tmp, tmp_device_mesh);

    return parameter_sync_cost(param_size_bytes, src, dst, cost_table);
  });

  m.def("mcmc_search_time_profile", &py_single_mcmc_search_time_profile);
};
