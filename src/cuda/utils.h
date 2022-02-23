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
                            + std::string(ctranslate2::cuda::cublasGetStatusName(status))); \
    }

    const char* cublasGetStatusName(cublasStatus_t status);

    cudaStream_t get_cuda_stream();
    cublasHandle_t get_cublas_handle();

    int get_gpu_count();
    bool has_gpu();
    const cudaDeviceProp& get_device_properties(int device = -1);
    bool gpu_supports_int8(int device = -1);
    bool gpu_has_int8_tensor_cores(int device = -1);
    bool gpu_has_fp16_tensor_cores(int device = -1);

// Convenience macro to call Thrust functions with a default execution policy.
#define THRUST_CALL(FUN, ...) FUN(thrust::cuda::par_nosync.on(ctranslate2::cuda::get_cuda_stream()), __VA_ARGS__)

  }
}
