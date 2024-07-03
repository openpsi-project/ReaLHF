#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<typename T>
__global__ void copyDataKernel(T *dst, const T *src, size_t *dst_offsets, size_t *src_offsets,
                               size_t *sizes, size_t N) {
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
      {
        size_t dst_offset = dst_offsets[i];
        size_t src_offset = src_offsets[i];
        size_t size = sizes[i];
        memcpy(dst + dst_offset, src + src_offset, size * sizeof(T));
      }
    }
  }
}

template<size_t N, typename T>
void set_intervals(const at::Tensor src, at::Tensor dst,
                   std::vector<std::pair<size_t, size_t>> intervals) {
  {
    TORCH_CHECK(src.dtype() == dst.dtype(),
                "Source and destination tensors must have the same dtype");
    TORCH_CHECK(intervals.size() == N, "Expected ", N, " tensors, but got ", intervals.size());
    TORCH_CHECK(src.ndimension() == 1, "Expected 1D tensor, but got ", src.ndimension());
    TORCH_CHECK(dst.ndimension() == 1, "Expected 1D tensor, but got ", dst.ndimension());
    TORCH_CHECK(src.is_cuda(), "Expected source tensor to be on CUDA, but got ", src.device());
    TORCH_CHECK(dst.is_cuda(), "Expected destination tensor to be on CUDA, but got ", dst.device());
    size_t src_size = src.size(0);
    size_t dst_size = dst.size(0);

    // Create helper arrays for indexing.
    // offsets is used to index into the source tensor.
    // interval_starts is used to index into the destination tensor.
    // interval_sizes is used to set the size in memcpy.
    std::vector<size_t> offsets(N), interval_sizes(N), interval_starts(N);
    size_t offset = 0;
    for (auto i = 0; i < N; i++) {
      {
        const auto &interval = intervals[i];
        TORCH_CHECK(interval.second <= dst_size,
                    "Interval end must be less than or equal to the size of the tensor, but got ",
                    interval.second);
        TORCH_CHECK(interval.second >= interval.first,
                    "Interval end must be greater than or equal to the start, but got ",
                    interval.first, " and ", interval.second);
        offsets[i] = offset;
        interval_sizes[i] = interval.second - interval.first;
        interval_starts[i] = interval.first;
        offset += interval_sizes[i];
      }
    }
    TORCH_CHECK(offset == src_size,
                "Sum of intervals must be equal to the size of the source tensor, but got ",
                offset);

    // Move helper arrays into GPU for the kernel.
    size_t *device_offsets, *device_interval_sizes, *device_interval_starts;
    cudaMalloc((void **)&device_offsets, N * sizeof(size_t));
    cudaMalloc((void **)&device_interval_sizes, N * sizeof(size_t));
    cudaMalloc((void **)&device_interval_starts, N * sizeof(size_t));
    cudaMemcpy(device_offsets, offsets.data(), N * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device_interval_sizes, interval_sizes.data(), N * sizeof(size_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_interval_starts, interval_starts.data(), N * sizeof(size_t),
               cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    copyDataKernel<T><<<numBlocks, blockSize>>>(dst.data_ptr<T>(), src.data_ptr<T>(),
                                                device_interval_starts, device_offsets,
                                                device_interval_sizes, N);

    // Synchronize CUDA threads to prevent errors.
    // Since the copy is usually fast, synchronization overhead is acceptable.
    cudaDeviceSynchronize();

    // Free allocated device memory
    cudaFree(device_offsets);
    cudaFree(device_interval_sizes);
    cudaFree(device_interval_starts);
  }
}