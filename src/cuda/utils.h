#pragma once

#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <NvInfer.h>

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

#define CUDNN_CHECK(ans)                                                \
    {                                                                   \
      cudnnStatus_t status = (ans);                                     \
      if (status != CUDNN_STATUS_SUCCESS)                               \
        THROW_RUNTIME_ERROR("cuDNN failed with status "                 \
                            + std::string(cudnnGetErrorString(status))); \
    }

// Default execution policy for Thrust.
#define THRUST_EXECUTION_POLICY thrust::cuda::par(ctranslate2::cuda::get_thrust_allocator()) \
    .on(ctranslate2::cuda::get_cuda_stream())

// Convenience macro to call Thrust functions with the default execution policy.
#define THRUST_CALL(FUN, ...) FUN(THRUST_EXECUTION_POLICY, __VA_ARGS__)

    std::string cublasGetStatusString(cublasStatus_t status);

    cudaStream_t get_cuda_stream();
    cublasHandle_t get_cublas_handle();
    cudnnHandle_t get_cudnn_handle();

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
    bool has_fast_fp16();
    bool has_fast_int8();

    // Custom allocator for Thrust.
    class ThrustAllocator {
    public:
      typedef char value_type;
      value_type* allocate(std::ptrdiff_t num_bytes);
      void deallocate(value_type* p, size_t);
    };

    ThrustAllocator& get_thrust_allocator();

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
      nvinfer1::ICudaEngine* _engine = nullptr;
      nvinfer1::IExecutionContext* _execution_context = nullptr;
    };

    // Statically assiocate cudnnDataType_t with a C++ type.
    template <class T>
    struct TypeToCUDNNType {};

#define MATCH_CUDNN_DATA_TYPE(TYPE, CUDNN_DATA_TYPE)            \
    template<>                                                  \
    struct TypeToCUDNNType<TYPE> {                              \
      static constexpr cudnnDataType_t value = CUDNN_DATA_TYPE; \
    }

    MATCH_CUDNN_DATA_TYPE(float, CUDNN_DATA_FLOAT);
    MATCH_CUDNN_DATA_TYPE(int8_t, CUDNN_DATA_INT8);
    MATCH_CUDNN_DATA_TYPE(int32_t, CUDNN_DATA_INT32);

  }
}
