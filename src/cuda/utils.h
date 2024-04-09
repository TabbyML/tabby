#pragma once

#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/execution_policy.h>

#ifdef CT2_WITH_TENSOR_PARALLEL
#  include <cuda/mpi_stub.h>
#  include <nccl.h>
#endif
#ifdef CT2_WITH_CUDNN
#  include <cudnn.h>
#endif

#include "ctranslate2/types.h"
#include "ctranslate2/utils.h"

namespace ctranslate2 {
  namespace cuda {

#ifdef CT2_WITH_TENSOR_PARALLEL
#define MPI_CHECK(ans)                                                  \
  {                                                                     \
    int e = ans;                                                        \
    if( e != MPI_SUCCESS )                                              \
      THROW_RUNTIME_ERROR("MPI failed with error "                      \
                          + std::to_string(e));                         \
  }

#define NCCL_CHECK(ans)                                                 \
  {                                                                     \
    ncclResult_t r = ans;                                               \
    if( r != ncclSuccess )                                              \
      THROW_RUNTIME_ERROR("NCCL failed with error "                     \
                          + std::to_string(r));                         \
  }
#endif

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

#define CUDNN_CHECK(ans)                                                \
    {                                                                   \
      cudnnStatus_t status = (ans);                                     \
      if (status != CUDNN_STATUS_SUCCESS)                               \
        THROW_RUNTIME_ERROR("cuDNN failed with status "                 \
                            + std::string(cudnnGetErrorString(status))); \
    }

#define TENSOR_CHECK(ans, message)                                      \
    {                                                                   \
      if (!ans)                                                 \
        THROW_RUNTIME_ERROR(message);                                   \
    }

    const char* cublasGetStatusName(cublasStatus_t status);

    cudaStream_t get_cuda_stream();
    cublasHandle_t get_cublas_handle();
#ifdef CT2_WITH_CUDNN
    cudnnHandle_t get_cudnn_handle();
    cudnnDataType_t get_cudnn_data_type(DataType dtype);
#endif

    int get_gpu_count();
    bool has_gpu();
    const cudaDeviceProp& get_device_properties(int device = -1);
    bool gpu_supports_int8(int device = -1);
    bool gpu_has_int8_tensor_cores(int device = -1);
    bool gpu_has_fp16_tensor_cores(int device = -1);
    bool have_same_compute_capability(const std::vector<int>& devices);

    bool use_true_fp16_gemm();
    void use_true_fp16_gemm(bool use);

    class UseTrueFp16GemmInScope {
    public:
      UseTrueFp16GemmInScope(const bool use)
        : _previous_value(use_true_fp16_gemm())
        , _scope_value(use)
      {
        use_true_fp16_gemm(_scope_value);
      }

      ~UseTrueFp16GemmInScope() {
        use_true_fp16_gemm(_previous_value);
      }

    private:
      const bool _previous_value;
      const bool _scope_value;
    };

// Convenience macro to call Thrust functions with a default execution policy.
#define THRUST_CALL(FUN, ...) FUN(thrust::cuda::par_nosync.on(ctranslate2::cuda::get_cuda_stream()), __VA_ARGS__)

  }
}
