#include "ctranslate2/cuda/utils.h"

#include <stdexcept>

namespace ctranslate2 {
  namespace cuda {

    static std::string cublasGetStatusString(cublasStatus_t status)
    {
      switch (status)
      {
      case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
      case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
      case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
      case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
      case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
      case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
      case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
      case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
      case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
      case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
      default:
        return "UNKNOWN";
      }
    }

    void cuda_assert(cudaError_t code, const std::string& file, int line)
    {
      if (code != cudaSuccess)
        throw std::runtime_error("CUDA failed with error '"
                                 + std::string(cudaGetErrorString(code))
                                 + "' at " + file + ":" + std::to_string(line));
    }

    void cublas_assert(cublasStatus_t status, const std::string& file, int line)
    {
      if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS failed with status '"
                                 + cublasGetStatusString(status)
                                 + "' at " + file + ":" + std::to_string(line));
    }

    void cudnn_assert(cudnnStatus_t code, const std::string& file, int line)
    {
      if (code != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error("cuDNN failed with error '"
                                 + std::string(cudnnGetErrorString(code))
                                 + "' at " + file + ":" + std::to_string(line));
    }

    cudaStream_t get_cuda_stream() {
      // Use one CUDA stream per host thread.
      static thread_local cudaStream_t stream;
      static thread_local bool initialized = false;
      if (!initialized) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        initialized = true;
      }
      return stream;
    }

    cublasHandle_t get_cublas_handle() {
      // Use one cuBLAS handle per host thread.
      static thread_local cublasHandle_t handle;
      static thread_local bool initialized = false;
      if (!initialized) {
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetStream(handle, get_cuda_stream()));
        initialized = true;
      }
      return handle;
    }

    cudnnHandle_t get_cudnn_handle() {
      // Use one cuDNN handle per host thread.
      static thread_local cudnnHandle_t handle;
      static thread_local bool initialized = false;
      if (!initialized) {
        CUDNN_CHECK(cudnnCreate(&handle));
        CUDNN_CHECK(cudnnSetStream(handle, get_cuda_stream()));
        initialized = true;
      }
      return handle;
    }

    int get_gpu_count() {
      int gpu_count = 0;
      cudaError_t status = cudaGetDeviceCount(&gpu_count);
      if (status != cudaSuccess)
        return 0;
      return gpu_count;
    }

  }
}
