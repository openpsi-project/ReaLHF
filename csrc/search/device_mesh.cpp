#include <device_mesh.hpp>
#include <cassert>
#include <iostream>

// DeviceMesh::DeviceMesh()
// : device_mesh_name(""), n_nodes(0), n_gpus(0), node_names({}), gpu_ids({}) {
// };

DeviceMesh::DeviceMesh(int n_nodes, int n_gpus_per_node, std::vector<std::vector<int>> mapping,
                       std::string global_mesh_name, std::string name)
    : n_nodes(n_nodes),
      n_gpus_per_node(n_gpus_per_node),
      mapping(mapping),
      global_mesh_name(global_mesh_name),
      name(name) {
  assert(n_nodes == static_cast<int>(mapping.size()));
  for (int i = 0; i < n_nodes; i++) {
    assert(n_gpus_per_node == static_cast<int>(mapping[i].size()));
  }
};

bool is_all_overlap(std::vector<DeviceMesh *> device_meshes, DeviceMesh device_mesh) {
  for (DeviceMesh *other : device_meshes) {
    if (!device_mesh.overlap(*other)) return false;
  }
  return true;
};

bool is_all_overlap(std::unordered_set<DeviceMesh *> device_meshes, DeviceMesh device_mesh) {
  for (DeviceMesh *other : device_meshes) {
    if (!device_mesh.overlap(*other)) return false;
  }
  return true;
};

bool DeviceMesh::contain(const DeviceMesh &other) {
  // check whether one device mapping is contained by another by
  // checking 1. whether global_mesh_name is identical
  // 2. whether mapping of one device mesh is contained by the other one
  if (global_mesh_name != other.global_mesh_name) return false;
  for (int i = 0; i < n_nodes; i++) {
    for (int j = 0; j < n_gpus_per_node; j++) {
      if (mapping[i][j] == 0 && other.mapping[i][j] == 1) return false;
    }
  }
  return true;
};

bool DeviceMesh::contained_by(const DeviceMesh &other) {
  if (global_mesh_name != other.global_mesh_name) return false;
  for (int i = 0; i < n_nodes; i++) {
    for (int j = 0; j < n_gpus_per_node; j++) {
      if (mapping[i][j] == 1 && other.mapping[i][j] == 0) return false;
    }
  }
  return true;
};

bool DeviceMesh::overlap(const DeviceMesh &other) {
  if (global_mesh_name != other.global_mesh_name) return false;
  for (int i = 0; i < n_nodes; i++) {
    for (int j = 0; j < n_gpus_per_node; j++) {
      if (mapping[i][j] == 1 && other.mapping[i][j] == 1) return true;
    }
  }
  return false;
};

ModelParallelStrategy::ModelParallelStrategy(int num_pp, int num_dp, int num_mp)
    : num_pp(num_pp), num_dp(num_dp), num_mp(num_mp) {};

bool ModelParallelStrategy::operator==(const ModelParallelStrategy &other) const {
  return num_pp == other.num_pp && num_dp == other.num_dp && num_mp == other.num_mp;
};

bool DeviceMesh::operator==(const DeviceMesh &other) const {
  return name == other.name && global_mesh_name == other.global_mesh_name;
};

std::string ModelParallelStrategy::to_string() {
  return "num_pp:" + std::to_string(num_pp) + ";" + "num_dp:" + std::to_string(num_dp) + ";"
         + "num_mp:" + std::to_string(num_mp);
};

std::string ModelParallelStrategy::to_key() {
  return std::to_string(num_pp) + "," + std::to_string(num_mp) + "," + std::to_string(num_dp);
}