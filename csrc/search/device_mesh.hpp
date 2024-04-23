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
    std::string device_mesh_name;
    int n_nodes;
    int n_gpus;
    std::unordered_set<std::string> node_names;
    std::unordered_set<int> gpu_ids;
    int n_gpus_per_node = 8;
    RPCInstance* pre_task = nullptr;
    
    // DeviceMesh();
    DeviceMesh(std::string device_mesh_name, 
               int n_nodes, 
               int n_gpus, 
               std::unordered_set<std::string> node_names, 
               std::unordered_set<int> gpu_ids, 
               int n_gpus_per_node);
    
    DeviceMesh(std::string device_mesh_name, 
               int n_nodes, 
               int n_gpus, 
               std::vector<std::string> node_names, 
               std::vector<int> gpu_ids, 
               int n_gpus_per_node);

    bool is_overlap(const DeviceMesh& other);
    bool is_contain(const DeviceMesh& other);

    bool operator==(const DeviceMesh& other) const;

};

std::vector<std::unordered_set<std::string>> divide(DeviceMesh& device_mesh, int n);

bool is_all_overlap(std::vector<DeviceMesh*> device_meshes, DeviceMesh device_mesh);
bool is_all_overlap(std::unordered_set<DeviceMesh*> device_meshes, DeviceMesh device_mesh);

class ModelParallelStrategy{
public:
    int num_pp, num_dp, num_mp;
    
    ModelParallelStrategy(int num_pp, int num_dp, int num_mp);

    bool operator==(const ModelParallelStrategy& other) const;

    std::string to_string();
    std::string to_key();
};

class ModelDeviceMapping {  
};



#endif // DEVICE_MESH_HPP