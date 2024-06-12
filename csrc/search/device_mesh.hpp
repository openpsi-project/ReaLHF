#ifndef DEVICE_MESH_HPP
#define DEVICE_MESH_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
// #include <rpc.hpp>

class RPCInstance;

class DeviceMesh {
 public:
  int n_nodes;
  int n_gpus_per_node;
  std::vector<std::vector<int>> mapping;
  std::string global_mesh_name;
  std::string name;
  RPCInstance *pre_task = nullptr;

  // DeviceMesh();
  DeviceMesh(int n_nodes, int n_gpus_per_node, std::vector<std::vector<int>> mapping,
             std::string global_mesh_name, std::string name);

  bool overlap(const DeviceMesh &other);
  bool contain(const DeviceMesh &other);
  bool contained_by(const DeviceMesh &other);

  bool operator==(const DeviceMesh &other) const;
};

bool is_all_overlap(std::vector<DeviceMesh *> device_meshes, DeviceMesh device_mesh);
bool is_all_overlap(std::unordered_set<DeviceMesh *> device_meshes, DeviceMesh device_mesh);

class ModelParallelStrategy {
 public:
  int num_pp, num_dp, num_mp;

  ModelParallelStrategy(int num_pp, int num_dp, int num_mp);

  bool operator==(const ModelParallelStrategy &other) const;

  std::string to_string();
  std::string to_key();
};

class ModelDeviceMapping {};

#endif  // DEVICE_MESH_HPP