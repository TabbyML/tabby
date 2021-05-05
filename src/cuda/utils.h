#pragma once

#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/execution_policy.h>

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

    std::string cublasGetStatusString(cublasStatus_t status);

    cudaStream_t get_cuda_stream();
    cublasHandle_t get_cublas_handle();

    int get_gpu_count();
    bool has_gpu();
    const cudaDeviceProp& get_device_properties(int device = -1);
    bool gpu_supports_int8(int device = -1);
    bool gpu_has_int8_tensor_cores(int device = -1);
    bool gpu_has_fp16_tensor_cores(int device = -1);

    // Define a custom execution policy to set the default stream and disable synchronization.
    struct thrust_execution_policy : thrust::device_execution_policy<thrust_execution_policy> {
    private:
      cudaStream_t _stream = get_cuda_stream();

      friend __host__ __device__ cudaStream_t get_stream(thrust_execution_policy& policy) {
        return policy._stream;
      }

      friend __host__ __device__ cudaError_t synchronize_stream(thrust_execution_policy&) {
        return cudaSuccess;
      }
    };

// Convenience macro to call Thrust functions with a default execution policy.
#define THRUST_CALL(FUN, ...) FUN(ctranslate2::cuda::thrust_execution_policy(), __VA_ARGS__)

  }
}
