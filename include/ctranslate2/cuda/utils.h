#pragma once

#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

namespace ctranslate2 {
  namespace cuda {

#define CUDA_CHECK(ans) { ctranslate2::cuda::cuda_assert((ans), __FILE__, __LINE__); }
#define CUBLAS_CHECK(ans) { ctranslate2::cuda::cublas_assert((ans), __FILE__, __LINE__); }
#define CUDNN_CHECK(ans) { ctranslate2::cuda::cudnn_assert((ans), __FILE__, __LINE__); }

    void cuda_assert(cudaError_t code, const std::string& file, int line);
    void cublas_assert(cublasStatus_t status, const std::string& file, int line);
    void cudnn_assert(cudnnStatus_t status, const std::string& file, int line);

    cudaStream_t& get_cuda_stream();
    cublasHandle_t& get_cublas_handle();
    cudnnHandle_t& get_cudnn_handle();

  }
}
