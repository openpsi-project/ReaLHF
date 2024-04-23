#include <torch/extension.h>
#include <cuda_fp16.h>

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...)                                   \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
              #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

template<typename scalar_t, int chunk_size>
__global__ void set_intervals_kernel(const scalar_t *src, scalar_t *dst, const long *interval_sizes,
                                     const long *intervals, const long *offsets,
                                     int num_intervals) {
  int interval_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (interval_id >= num_intervals) { return; }
  long interval_size = interval_sizes[interval_id];
  long offset = offsets[interval_id];
  long start_idx = intervals[2 * interval_id];
  int tid = blockIdx.y * blockDim.y + threadIdx.y;
  for (int i = 0; i < chunk_size; ++i) {
    if (tid * chunk_size + i >= interval_size) { break; }
    dst[start_idx + tid * chunk_size + i] = src[offset + tid * chunk_size + i];
  }
}

template<typename scalar_t, int chunk_size>
void set_intervals_cuda(const torch::Tensor &src, const torch::Tensor &dst,
                        const torch::Tensor &intervals, const torch::Tensor &interval_sizes,
                        const torch::Tensor &offsets, long max_interval_size) {
  CHECK_DEVICE(src);
  CHECK_DEVICE(dst);
  CHECK_DEVICE(intervals);
  CHECK_DEVICE(interval_sizes);
  CHECK_DEVICE(offsets);
  CHECK_CONTIGUOUS(src);
  CHECK_CONTIGUOUS(dst);
  CHECK_CONTIGUOUS(intervals);
  CHECK_CONTIGUOUS(interval_sizes);
  CHECK_CONTIGUOUS(offsets);
  int num_intervals = intervals.size(0);
  TORCH_CHECK(intervals.dtype() == torch::kInt64, "intervals must be of type long");
  TORCH_CHECK(interval_sizes.dtype() == torch::kInt64, "interval_sizes must be of type long");
  TORCH_CHECK(offsets.dtype() == torch::kInt64, "offsets must be of type long");
  CHECK_SHAPE(interval_sizes, num_intervals);
  CHECK_SHAPE(offsets, num_intervals);

  // Launch kernel
  const int threads_per_block_x = 32;
  const int threads_per_block_y = 32;

  const int num_blocks_x = (num_intervals + threads_per_block_x - 1) / threads_per_block_x;

  const int n_chunks = (max_interval_size + chunk_size - 1) / chunk_size;
  const int num_blocks_y = (n_chunks + threads_per_block_y - 1) / threads_per_block_y;

  const dim3 numBlocks(num_blocks_x, num_blocks_y);
  const dim3 threadsPerBlock(threads_per_block_x, threads_per_block_y);

  set_intervals_kernel<scalar_t, chunk_size><<<numBlocks, threadsPerBlock>>>(
      src.data_ptr<scalar_t>(), dst.data_ptr<scalar_t>(), interval_sizes.data_ptr<long>(),
      intervals.data_ptr<long>(), offsets.data_ptr<long>(), num_intervals);
}

template<typename scalar_t, int chunk_size>
__global__ void slice_intervals_kernel(const scalar_t *input_data, scalar_t *output_data,
                                       const long *interval_sizes, const long *intervals,
                                       const long *offsets, int num_intervals) {
  int interval_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (interval_id >= num_intervals) { return; }
  long interval_size = interval_sizes[interval_id];
  long offset = offsets[interval_id];
  long start_idx = intervals[2 * interval_id];
  int tid = blockIdx.y * blockDim.y + threadIdx.y;
  for (int i = 0; i < chunk_size; ++i) {
    if (tid * chunk_size + i >= interval_size) { break; }
    output_data[offset + tid * chunk_size + i] = input_data[start_idx + tid * chunk_size + i];
  }
}

template<typename scalar_t, int chunk_size>
torch::Tensor slice_intervals_cuda(const torch::Tensor &input, const torch::Tensor &intervals,
                                   const torch::Tensor &interval_sizes,
                                   const torch::Tensor &offsets, long max_interval_size,
                                   long output_size) {
  CHECK_DEVICE(input);
  CHECK_DEVICE(intervals);
  CHECK_DEVICE(interval_sizes);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(intervals);
  CHECK_CONTIGUOUS(interval_sizes);
  int num_intervals = intervals.size(0);
  TORCH_CHECK(intervals.dtype() == torch::kInt64, "intervals must be of type long");
  TORCH_CHECK(interval_sizes.dtype() == torch::kInt64, "interval_sizes must be of type long");
  TORCH_CHECK(offsets.dtype() == torch::kInt64, "offsets must be of type long");
  CHECK_SHAPE(interval_sizes, num_intervals);
  CHECK_SHAPE(offsets, num_intervals);

  // Create output tensor
  torch::Tensor output = torch::empty(output_size, input.options());

  // Launch kernel
  const int threads_per_block_x = 32;
  const int threads_per_block_y = 32;

  const int num_blocks_x = (num_intervals + threads_per_block_x - 1) / threads_per_block_x;

  const int n_chunks = (max_interval_size + chunk_size - 1) / chunk_size;
  const int num_blocks_y = (n_chunks + threads_per_block_y - 1) / threads_per_block_y;

  const dim3 numBlocks(num_blocks_x, num_blocks_y);
  const dim3 threadsPerBlock(threads_per_block_x, threads_per_block_y);

  slice_intervals_kernel<scalar_t, chunk_size><<<numBlocks, threadsPerBlock>>>(
      input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), interval_sizes.data_ptr<long>(),
      intervals.data_ptr<long>(), offsets.data_ptr<long>(), num_intervals);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("set_intervals_cuda", &set_intervals_cuda<float, 2048>, "Slice intervals CUDA");
  m.def("set_intervals_cuda_half", &set_intervals_cuda<at::Half, 2048>,
        "Slice intervals CUDA (float16)");
  m.def("set_intervals_cuda_bfloat16", &set_intervals_cuda<at::BFloat16, 2048>,
        "Slice intervals CUDA (bfloat16)");
  m.def("slice_intervals_cuda", &slice_intervals_cuda<float, 2048>, "Slice intervals CUDA");
  m.def("slice_intervals_cuda_half", &slice_intervals_cuda<at::Half, 2048>,
        "Slice intervals CUDA (float16)");
  m.def("slice_intervals_cuda_bfloat16", &slice_intervals_cuda<at::BFloat16, 2048>,
        "Slice intervals CUDA (bfloat16)");
}