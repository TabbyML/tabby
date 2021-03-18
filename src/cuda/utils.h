#pragma once

#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "ctranslate2/types.h"
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

    int get_gpu_count();
    bool has_gpu();
    const cudaDeviceProp& get_device_properties(int device = -1);
    bool gpu_supports_int8(int device = -1);
    bool gpu_has_int8_tensor_cores(int device = -1);
    bool gpu_has_fp16_tensor_cores(int device = -1);

    // Custom allocator for Thrust.
    class ThrustAllocator {
    public:
      typedef char value_type;
      value_type* allocate(std::ptrdiff_t num_bytes);
      void deallocate(value_type* p, size_t);
    };

    ThrustAllocator& get_thrust_allocator();

  }
}
