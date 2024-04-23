#include <device_mesh.hpp>
#include <cassert>
#include <iostream>

// DeviceMesh::DeviceMesh()
// : device_mesh_name(""), n_nodes(0), n_gpus(0), node_names({}), gpu_ids({}) {
// };

DeviceMesh::DeviceMesh(std::string device_mesh_name, int n_nodes, int n_gpus,
                       std::unordered_set<std::string> node_names, std::unordered_set<int> gpu_ids,
                       int n_gpus_per_node)
    : device_mesh_name(device_mesh_name),
      n_nodes(n_nodes),
      n_gpus(n_gpus),
      node_names(node_names),
      gpu_ids(gpu_ids),
      n_gpus_per_node(n_gpus_per_node) {
  assert(n_nodes == static_cast<int>(node_names.size()));
  assert(n_gpus == static_cast<int>(gpu_ids.size()) * n_nodes);
};

DeviceMesh::DeviceMesh(std::string device_mesh_name, int n_nodes, int n_gpus,
                       std::vector<std::string> node_names, std::vector<int> gpu_ids,
                       int n_gpus_per_node)
    : device_mesh_name(device_mesh_name),
      n_nodes(n_nodes),
      n_gpus(n_gpus),
      n_gpus_per_node(n_gpus_per_node) {
  assert(n_nodes == static_cast<int>(node_names.size()));
  assert(n_gpus == static_cast<int>(gpu_ids.size()) * n_nodes);
  for (std::string node_name : node_names) { this->node_names.insert(node_name); }
  for (int gpu_id : gpu_ids) { this->gpu_ids.insert(gpu_id); }
};

bool DeviceMesh::is_overlap(const DeviceMesh &other) {
  if (n_nodes > 1) {
    for (std::string node_name : node_names) {
      if (other.node_names.find(node_name) != other.node_names.end()) return true;
    }
  } else if (n_nodes == 1) {
    if (other.node_names.find(*node_names.begin()) == other.node_names.end()) return false;
    for (int gpu_id : gpu_ids) {
      if (other.gpu_ids.find(gpu_id) != other.gpu_ids.end()) return true;
    }
  }
  return false;
};

bool is_all_overlap(std::vector<DeviceMesh *> device_meshes, DeviceMesh device_mesh) {
  for (DeviceMesh *other : device_meshes) {
    if (!device_mesh.is_overlap(*other)) return false;
  }
  return true;
};

bool is_all_overlap(std::unordered_set<DeviceMesh *> device_meshes, DeviceMesh device_mesh) {
  for (DeviceMesh *other : device_meshes) {
    if (!device_mesh.is_overlap(*other)) return false;
  }
  return true;
};

bool DeviceMesh::is_contain(const DeviceMesh &other) {
  if (n_nodes > 1) {
    for (std::string node_name : other.node_names) {
      if (node_names.find(node_name) == node_names.end()) return false;
    }
  } else if (n_nodes == 1) {
    if (other.node_names.find(*node_names.begin()) == other.node_names.end()) return false;
    for (int gpu_id : other.gpu_ids) {
      if (gpu_ids.find(gpu_id) == gpu_ids.end()) return false;
    }
  }
  return true;
};

std::vector<std::unordered_set<std::string>> divide(DeviceMesh &device_mesh, int n) {
  std::vector<std::unordered_set<std::string>> node_name_list;
  // std::cout << "device mesh name " << device_mesh.device_mesh_name
  //           << " n " << n << std::endl;

  node_name_list.resize(n);
  if (n == 1) {
    node_name_list[0] = device_mesh.node_names;
  } else {
    if (device_mesh.n_nodes % n == 0) {
      int n_nodes_per_group = device_mesh.n_nodes / n;
      int i = 0;
      for (std::string node_name : device_mesh.node_names) {
        // std::cout << node_name << std::endl;
        // std::cout << i / n_nodes_per_group << std::endl;
        node_name_list[i / n_nodes_per_group].insert(node_name);
        i++;
      }
    } else if (n % device_mesh.n_nodes == 0) {
      int n_groups_per_node = n / device_mesh.n_nodes;
      int i = 0;
      for (std::string node_name : device_mesh.node_names) {
        for (int j = 0; j < n_groups_per_node; j++) {
          // std::cout << node_name << std::endl;
          // std::cout << i * n_groups_per_node + j << std::endl;
          node_name_list[i * n_groups_per_node + j].insert(node_name);
        }
        i++;
      }
    } else {
      std::cout << "Divide fail: device mesh name " << device_mesh.device_mesh_name << " n " << n
                << std::endl;
      assert(false);
    }
  }
  return node_name_list;
};

ModelParallelStrategy::ModelParallelStrategy(int num_pp, int num_dp, int num_mp)
    : num_pp(num_pp), num_dp(num_dp), num_mp(num_mp) {};

bool ModelParallelStrategy::operator==(const ModelParallelStrategy &other) const {
  return num_pp == other.num_pp && num_dp == other.num_dp && num_mp == other.num_mp;
};

bool DeviceMesh::operator==(const DeviceMesh &other) const {
  return device_mesh_name == other.device_mesh_name;
};

std::string ModelParallelStrategy::to_string() {
  return "num_pp:" + std::to_string(num_pp) + ";" + "num_dp:" + std::to_string(num_dp) + ";"
         + "num_mp:" + std::to_string(num_mp);
};

std::string ModelParallelStrategy::to_key() {
  return std::to_string(num_pp) + "," + std::to_string(num_mp) + "," + std::to_string(num_dp);
}