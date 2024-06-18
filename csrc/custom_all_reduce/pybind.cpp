/* Copied from the vLLM project: https://github.com/vllm-project/vllm */
#include <torch/extension.h>

using fptr_t = uint64_t;
fptr_t init_custom_ar(torch::Tensor &meta, torch::Tensor &rank_data,
                      const std::vector<std::string> &handles, const std::vector<int64_t> &offsets,
                      int rank, bool full_nvlink);
bool should_custom_ar(torch::Tensor &inp, int max_size, int world_size, bool full_nvlink);
void all_reduce_reg(fptr_t _fa, torch::Tensor &inp, torch::Tensor &out);
void all_reduce_unreg(fptr_t _fa, torch::Tensor &inp, torch::Tensor &reg_buffer,
                      torch::Tensor &out);
void dispose(fptr_t _fa);
int meta_size();
void register_buffer(fptr_t _fa, torch::Tensor &t, const std::vector<std::string> &handles,
                     const std::vector<int64_t> &offsets);
std::pair<std::vector<uint8_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(fptr_t _fa);
void register_graph_buffers(fptr_t _fa, const std::vector<std::string> &handles,
                            const std::vector<std::vector<int64_t>> &offsets);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // vLLM custom all-reduce kernels
  pybind11::module custom_ar = m.def_submodule("custom_ar", "custom allreduce");
  custom_ar.def("init_custom_ar", &init_custom_ar, "init_custom_ar");
  custom_ar.def("should_custom_ar", &should_custom_ar, "should_custom_ar");
  custom_ar.def("all_reduce_reg", &all_reduce_reg, "all_reduce_reg");
  custom_ar.def("all_reduce_unreg", &all_reduce_unreg, "all_reduce_unreg");
  custom_ar.def("dispose", &dispose, "dispose");
  custom_ar.def("meta_size", &meta_size, "meta_size");
  custom_ar.def("register_buffer", &register_buffer, "register_buffer");
  custom_ar.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta,
                "get_graph_buffer_ipc_meta");
  custom_ar.def("register_graph_buffers", &register_graph_buffers, "register_graph_buffers");
}
