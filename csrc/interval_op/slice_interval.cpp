#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>

template<size_t N>
at::Tensor slice_intervals(const at::Tensor &src,
                           const std::vector<std::pair<size_t, size_t>> &intervals) {
  {
    TORCH_CHECK(intervals.size() == N, "Expected ", N, " tensors, but got ", intervals.size());
    TORCH_CHECK(src.ndimension() == 1, "Expected 1D tensor, but got ", src.ndimension());
    const size_t src_size = src.size(0);

    auto offset = 0;
    for (auto i = 0; i < N; i++) {
      {
        const auto &interval = intervals[i];
        TORCH_CHECK(interval.second <= src_size,
                    "Interval end must be less than or equal to the size of the tensor, but got ",
                    interval.second);
        TORCH_CHECK(interval.second >= interval.first,
                    "Interval end must be greater than or equal to the start, but got ",
                    interval.first, " and ", interval.second);
        offset += interval.second - interval.first;
      }
    }

    // Prepare vector to hold slices
    at::Tensor out = at::empty({{offset}}, src.options());
    std::vector<at::Tensor> slices(N);

    // Parallel slicing
    // Parallel copy or dataCopyKernel is not as efficient as the at::cat_out function.
    at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
      {
        for (int64_t i = start; i < end; ++i) {
          {
            const auto &interval = intervals[i];
            slices[i] = src.slice(0, interval.first, interval.second);
          }
        }
      }
    });
    return at::cat_out(out, slices, 0);
  }
}