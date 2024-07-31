// Copied and modified from
// https://github.com/pytorch/pytorch/blob/v2.3.1/aten/src/ATen/cuda/CUDAGraph.cpp This
// implementation enables efficient graph recapture, which is used to save memory without degrading
// performance. When a graph is used periodically, its memory could not be released when the graph
// is unused. Then to save memory, this graph should be destroyed and recaptured. However in
// original pytorch implementation, recapturing the graph is expensive. This implementation uses
// `cudaGraphExecUpdate` to recapture the graph efficiently. NOTE:
// 1. This extension could only be compiled under torch version 2.3.1.
//    Otherwise compile may fail due to incompatible APIs in pytorch source code.
// 2. Compared to original pytorch CUDAGraph implementation, some features such as memory pool
// sharing is disabled
//    in this extension.
//    It is also unsafe to mix the usage of this and the original pytorch one.
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>

#include <chrono>
#include <thread>

#include "custom_cudagraph.h"

namespace at::cuda {

constexpr int kSynchronizeBusyWaitMillis = 10;

MempoolId_t graph_pool_handle() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  // uuid count starts at 1. 0 is reserved to mean "wasn't set by graph_pool_handle".
  static std::atomic<CaptureId_t> uid{1};
  // Sets just the second value, to distinguish it from MempoolId_ts created from
  // cudaStreamGetCaptureInfo id_s in capture_begin.
  return {0, uid++};
#else
  TORCH_CHECK(false,
              "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3")
  return {0, 0};
#endif
}

// Get the expected id of a capture sequence so that we can call beginAllocateStreamToPool
// before starting a graph capture
CaptureId_t capture_sequence_id() {
  // id starts at 1:
  // Ensures uuid count starts at 1. 0 is reserved to mean "not set by cudaStreamGetCaptureInfo".
  // (But how do we know GetCaptureInfo never sets id_ to 0? Because that's the current behavior,
  // and I asked cuda devs to keep it that way, and they agreed.)
  static std::atomic<CaptureId_t> uuid{1};
  return uuid++;
}

/**
 * Note [CUDA Graph Wrapper Class]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Q: Why do we need graph capture and launch bindings in Pytorch?
 *    Why can't they live in a user extension, for example?
 *
 * A1: Convenience.
 * A2: To ensure valid numerics on replay, some native CUDA ops (like RNG ops with
 *     CPU statefulness) need cooperation from the capture and replay bindings
 *     (see Note [CUDA Graph-safe RNG states] in CUDAGeneratorImpl.h).
 *
 *     We can't expect users to know about this cooperation.  If users write capture
 *     bindings naively in an extension, they likely won't interact with the native
 *     ops properly.  Their graphs would yield invalid numerics on replay.
 */

/**
 * Note [Interaction with CUDA graph capture] in CUDACachingAllocator.cpp
 * describes memory management for captures.
 */

std::atomic<int> CustomizedCUDAGraph::pending_event_queries = 0;

// Track any outstanding event queries that could happen e.g., in a NCCL watchdog so that they
// can be resolved before the capture begins. Note that event queries are not allowed during a
// graph capture in the default capture mode.
void CustomizedCUDAGraph::inc_pending_event_queries() { pending_event_queries++; }

void CustomizedCUDAGraph::dec_pending_event_queries() {
  TORCH_INTERNAL_ASSERT(
      pending_event_queries > 0,
      "Attempted to decrement the number of outstanding events to be queried, but it was <= 0.");
  pending_event_queries--;
}

int CustomizedCUDAGraph::num_pending_event_queries() { return pending_event_queries; }

CustomizedCUDAGraph::CustomizedCUDAGraph()
    // CUDAStreams may not be default-constructed.
    : capture_stream_(at::cuda::getCurrentCUDAStream()) {
#if (defined(USE_ROCM) && ROCM_VERSION < 50300)
  TORCH_CHECK(false,
              "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3");
#endif
}

void CustomizedCUDAGraph::capture_begin(
    // MempoolId_t pool/*=0*/, cudaStreamCaptureMode capture_mode
) {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  MempoolId_t pool = {0, 0};
  cudaStreamCaptureMode capture_mode = cudaStreamCaptureModeGlobal;
  //   TORCH_CHECK(!has_graph_exec_,
  //               "This CustomizedCUDAGraph instance already owns a captured graph. "
  //               "To capture a new graph, create a new instance.");

  // For now, a CUDAGraph instance only accommodates the default generator on the device that's
  // current when capture begins. If any op in the captured region uses a non-default generator,
  // or a generator on another device, the offending generator will throw an error.
  // These restrictions simplify CUDAGraph, but could be relaxed in the future:
  // in principle, the underlying Cuda calls do permit cross-device ops to be captured.
  auto *gen = get_generator_or_default<CUDAGeneratorImpl>(c10::nullopt,
                                                          cuda::detail::getDefaultCUDAGenerator());

  auto options = TensorOptions().device(at::kCUDA).dtype(at::kLong);
  seed_extragraph_ = at::empty({1}, options);
  offset_extragraph_ = at::empty({1}, options);

  seed_extragraph_.fill_(int64_t(gen->current_seed()));
  gen->capture_prologue(seed_extragraph_.data_ptr<int64_t>(),
                        offset_extragraph_.mutable_data_ptr<int64_t>());

  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream != at::cuda::getDefaultCUDAStream(),
              "CUDA graphs must be captured on a non-default stream. "
              "(However, after capture, it's ok to replay them on the "
              "default stream.)");

  capture_stream_ = stream;
  capture_gen_ = gen;
  capture_dev_ = c10::cuda::current_device();

  id_ = capture_sequence_id();

  if (pool.first != 0 || pool.second != 0) {
    // Either value being nonzero means the user supplied a pool to share.
    // But only one should be nonzero.
    // If pool was created by another graph's capture_begin, first should be nonzero.
    // If pool was created by graph_pool_handle, second should be nonzero.
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    mempool_id_ = pool;
  } else {
    // User did not ask us to share a mempool. Use our own id_ as our mempool_id_.
    // Sets just the first value, to distinguish it from MempoolId_ts created by
    // graph_pool_handle().
    mempool_id_ = {id_, 0};
  }

  // Addendum: beginAllocateStreamToPool is now called before cudaStreamBeginCapture to prevent an
  // autograd thread's free() call triggering an invalid cudaEventRecord in the caching allocator
  // due to the capture status being updated _after_ a capture had already started.
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      capture_dev_, mempool_id_, [this](cudaStream_t stream) {
        cudaStreamCaptureStatus status;
        CaptureId_t stream_capture_id;
        AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &stream_capture_id));
        return status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive
               && stream_capture_id == capture_id_;
      });

  // At this point, any NCCL watchdogs should be aware that we are in capture mode
  // and therefore should not enqueue any additional work that could be event-queried.
  // We still must wait on any existing work that has not been cleaned up.
  while (num_pending_event_queries()) {
    TORCH_WARN_ONCE("Waiting for pending NCCL work to finish before starting graph capture.");
    std::this_thread::sleep_for(std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
  }

  // cudaStreamCaptureModeGlobal is the most conservative option to
  // prevent potentially unsafe CUDA API calls during capture.  See
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85
  AT_CUDA_CHECK(cudaStreamBeginCapture(capture_stream_, capture_mode));

  cudaStreamCaptureStatus status;
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &capture_id_));
  TORCH_INTERNAL_ASSERT(status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive);

  TORCH_INTERNAL_ASSERT(id_ > 0);
#else
  TORCH_CHECK(false,
              "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3")
#endif
}

void CustomizedCUDAGraph::capture_end() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream == capture_stream_, "Capture must end on the same stream it began on.");

  AT_CUDA_CHECK(cudaStreamEndCapture(capture_stream_, &graph_));

  c10::cuda::CUDACachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);

  TORCH_CHECK(graph_ != NULL, "Invalid capture.");

  auto *gen = get_generator_or_default<CUDAGeneratorImpl>(c10::nullopt,
                                                          cuda::detail::getDefaultCUDAGenerator());
  TORCH_CHECK(gen == capture_gen_, "Default CUDA RNG generator on current device at capture end "
                                   "is different from default generator on current device "
                                   "when capture began");
  wholegraph_increment_ = gen->capture_epilogue();

  size_t numCUDAGraphNodes = 0;
  AT_CUDA_CHECK(cudaGraphGetNodes(graph_, NULL, &numCUDAGraphNodes));
  if (numCUDAGraphNodes == 0) {
    TORCH_WARN("The CUDA Graph is empty. This usually means that the graph was ",
               "attempted to be captured on wrong device or stream.");
  }

  has_graph_ = true;
#else
  TORCH_CHECK(false,
              "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3")
#endif
}

void CustomizedCUDAGraph::instantiate() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  // In typical graph usage some tensors (e.g. the tensors used for graph IO) are not freed
  // between replays.
  // If Pytorch compiles and runs with a CUDA 11.4+ toolkit, there's a chance the allocator backend
  // is cudaMallocAsync.
  // cudaMallocAsync is generally graph-safe, but if some tensors are not freed between replays,
  // the graph's internal bookkeeping requires that we instantiate with
  // cudaGraphInstantiateFlagAutoFreeOnLaunch. See
  // cudaGraphLaunch
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1accfe1da0c605a577c22d9751a09597
  // cudaGraphInstantiateWithFlags
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1ga2c652a24ba93e52b99a47bec0888233
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11040)
  int version;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
#endif
    // Trailing NULL, NULL, 0 arguments were recommended by Cuda driver people,
    // who prefer not to report error message through these arguments moving forward
    // (they prefer return value, or errors on api calls internal to the capture)
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
    AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, 0));
#else
  AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
#endif
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11040)
  } else {
    AT_CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec_, graph_,
                                                cudaGraphInstantiateFlagAutoFreeOnLaunch));
  }
#endif

  has_graph_exec_ = true;

#else
  TORCH_CHECK(false,
              "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3")
#endif
}

void CustomizedCUDAGraph::destroy() {
// destroy CUDAGraph instance but not CUDAGraphExec
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  TORCH_CHECK(has_graph_,
              "Called CustomizedCUDAGraph::destroy without a preceding successful capture."
              "Nothing to destroy.");

  c10::cuda::CUDACachingAllocator::releasePool(capture_dev_, mempool_id_);
  C10_CUDA_CHECK_WARN(cudaGraphDestroy(graph_));
  has_graph_ = false;
#else
  TORCH_CHECK(false, "CUDA graphs is not yet supported on ROCM");
#endif
}

void CustomizedCUDAGraph::update() {
// Update CUDAGraphExec with newly captured CUDAGraph, to update memory pointers but retain graph
// topology.
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  TORCH_CHECK(has_graph_, "Called CustomizedCUDAGraph::update without a newly captured CUDAGraph.");
  TORCH_CHECK(has_graph_exec_,
              "Called CustomizedCUDAGraph::update without a instantiated CUDAGraphExec.");
  cudaGraphExecUpdateResultInfo updateResultInfo;
  cudaGraphExecUpdate(graph_exec_, graph_, &updateResultInfo);

  // if update failed, just re-instantiate the graph
  if (updateResultInfo.result != cudaGraphExecUpdateSuccess) {
    TORCH_WARN("CUDA Graph update failed, re-instantiating the graph.");
    C10_CUDA_CHECK_WARN(cudaGraphExecDestroy(graph_exec_));
    has_graph_exec_ = false;
    CustomizedCUDAGraph::instantiate();
  }
#else
  TORCH_CHECK(false, "CUDA graphs is not yet supported on ROCM");
#endif
}

void CustomizedCUDAGraph::replay() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  TORCH_CHECK(has_graph_exec_, "Called CUDAGraph::replay without a preceding successful capture.");

  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

  // Just like any RNG consumer kernel!
  auto *gen = get_generator_or_default<CUDAGeneratorImpl>(c10::nullopt,
                                                          cuda::detail::getDefaultCUDAGenerator());
  PhiloxCudaState rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(wholegraph_increment_);
  }
  seed_extragraph_.fill_(int64_t(gen->current_seed()));
  offset_extragraph_.fill_(int64_t(rng_engine_inputs.offset_.val));

  // graph_exec_ may be replayed in any stream.
  AT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, at::cuda::getCurrentCUDAStream()));

  int version;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
    // Workaround for bug in libcuda.so that causes replayed graphs with
    // certain topologies to be corrupted (kernels elided, internal syncs
    // ignored) when replayed back to back without a sync in between.
    // The bug is fixed in CUDA 11.4+.
    AT_CUDA_CHECK(cudaDeviceSynchronize());
  }
#else
  TORCH_CHECK(false, "CUDA graphs is not yet supported on ROCM");
#endif
}

void CustomizedCUDAGraph::reset() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  // I'd prefer these checks throw exceptions, not print warnings,
  // but the destructor calls reset(), and at least one CI build
  // refuses to compile with a throwing destructor.
  //
  // Instead of calling reset() in the destructor to clean up, I could
  // call reset() in the __del__ method of a thin Python wrapper,
  // in which case reset would be allowed to throw exceptions.
  // But Stackoverflow does not like user-defined __del__.
  // __del__ prevents Graph instances from EVER being garbage collected
  // if they participate in a reference cycle.
  // And exceptions thrown in __del__ only print a warning anyway.
  //
  // Calling reset() in the C++ destructor, with warnings instead of exceptions
  // if calls fail, is the compromise we chose.
  //
  // If capture_begin, the capture, or capture_end failed at some point, this CUDAGraph, the
  // generator, and the allocator could end up in all kinds of weird states depending where failure
  // occurred. If the user catches the failure exception in a script, or is running in REPL or (god
  // forbid) a Jupyter notebook, I don't see an easy way for reset() to gracefully fix all such
  // possible error states.
  if (has_graph_) {
    // notifyCaptureDestroy may throw. How should we handle this?
    c10::cuda::CUDACachingAllocator::releasePool(capture_dev_, mempool_id_);
    C10_CUDA_CHECK_WARN(cudaGraphDestroy(graph_));
    has_graph_ = false;
  }
  if (has_graph_exec_) {
    C10_CUDA_CHECK_WARN(cudaGraphExecDestroy(graph_exec_));
    has_graph_exec_ = false;
  }
#else
  TORCH_CHECK(false,
              "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3")
#endif
}

// Returns an id another graph's capture_begin can use to share the same memory pool as this graph.
MempoolId_t CustomizedCUDAGraph::pool() {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  TORCH_CHECK(has_graph_exec_, "Called CUDAGraph::pool() without a preceding successful capture.");
#else
  TORCH_CHECK(false,
              "CUDA graphs may only be used in Pytorch built with CUDA >= 11.0 or ROCM >= 5.3")
#endif
  return mempool_id_;
}

CustomizedCUDAGraph::~CustomizedCUDAGraph() { reset(); }

}  // namespace at::cuda
