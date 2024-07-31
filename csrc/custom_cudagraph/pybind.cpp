#include <torch/extension.h>
#include "custom_cudagraph.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<at::cuda::CustomizedCUDAGraph>(m, "CustomizedCUDAGraph")
      .def(py::init<>())
      .def("capture_begin", &at::cuda::CustomizedCUDAGraph::capture_begin)
      .def("capture_end", &at::cuda::CustomizedCUDAGraph::capture_end)
      .def("replay", &at::cuda::CustomizedCUDAGraph::replay)
      .def("reset", &at::cuda::CustomizedCUDAGraph::reset)
      .def("instantiate", &at::cuda::CustomizedCUDAGraph::instantiate)
      .def("destroy", &at::cuda::CustomizedCUDAGraph::destroy)
      .def("update", &at::cuda::CustomizedCUDAGraph::update);
};
