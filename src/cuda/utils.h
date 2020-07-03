#pragma once

#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef CT2_WITH_TENSORRT
#  include <NvInfer.h>
#endif

#include "ctranslate2/utils.h"

namespace ctranslate2 {
  namespace cuda {

#define CUDA_CHECK(ans)                                                 \
    {                                                                   \
      cudaError_t code = (ans);                                         \
      if (code != cudaSuccess)                                          \
        THROW_RUNTIME_ERROR("CUDA failed with error "                   \
                            + std::string(cudaGetErrorString(code)));   \
    }

#define CUBLAS_CHECK(ans)                                               \
    {                                                                   \
      cublasStatus_t status = (ans);                                    \
      if (status != CUBLAS_STATUS_SUCCESS)                              \
        THROW_RUNTIME_ERROR("cuBLAS failed with status "                \
                            + ctranslate2::cuda::cublasGetStatusString(status)); \
    }

// Default execution policy for Thrust.
#define THRUST_EXECUTION_POLICY thrust::cuda::par(ctranslate2::cuda::get_thrust_allocator()) \
    .on(ctranslate2::cuda::get_cuda_stream())

// Convenience macro to call Thrust functions with the default execution policy.
#define THRUST_CALL(FUN, ...) FUN(THRUST_EXECUTION_POLICY, __VA_ARGS__)

    std::string cublasGetStatusString(cublasStatus_t status);

    cudaStream_t get_cuda_stream();
    cublasHandle_t get_cublas_handle();

    // See https://nvlabs.github.io/cub/structcub_1_1_caching_device_allocator.html.
    struct CachingAllocatorConfig {
      unsigned int bin_growth = 4;
      unsigned int min_bin = 3;
      unsigned int max_bin = 12;
      size_t max_cached_bytes = 200 * (1 << 20);  // 200MB
    };

    CachingAllocatorConfig get_caching_allocator_config();

    int get_gpu_count();
    bool has_gpu();
    bool has_fast_int8(int device = -1);

    // Custom allocator for Thrust.
    class ThrustAllocator {
    public:
      typedef char value_type;
      value_type* allocate(std::ptrdiff_t num_bytes);
      void deallocate(value_type* p, size_t);
    };

    ThrustAllocator& get_thrust_allocator();

#ifdef CT2_WITH_TENSORRT
    class TensorRTLayer {
    public:
      virtual ~TensorRTLayer();

    protected:
      void run(void** bindings, const std::vector<nvinfer1::Dims>& input_dims);

      // These methods are called on the first call to run().
      virtual void build_network(nvinfer1::INetworkDefinition* network) = 0;
      virtual void set_optimization_profile(nvinfer1::IOptimizationProfile* profile) = 0;

    private:
      void build();
      int _device = 0;
      nvinfer1::ICudaEngine* _engine = nullptr;
      nvinfer1::IExecutionContext* _execution_context = nullptr;
    };
#endif

  }
}
