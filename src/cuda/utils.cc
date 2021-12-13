#include "./utils.h"

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#include "ctranslate2/utils.h"

namespace ctranslate2 {
  namespace cuda {

    const char* cublasGetStatusName(cublasStatus_t status)
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

    class CudaContext {
    public:
      CudaContext() {
        CUDA_CHECK(cudaGetDevice(&_device));
        CUDA_CHECK(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking));
        CUBLAS_CHECK(cublasCreate(&_handle));
        CUBLAS_CHECK(cublasSetStream(_handle, _stream));
      }
      ~CudaContext() {
        ScopedDeviceSetter scoped_device_setter(Device::CUDA, _device);
        cublasDestroy(_handle);
        cudaStreamDestroy(_stream);
      }
      cudaStream_t cuda_stream() const {
        return _stream;
      }
      cublasHandle_t cublas_handle() const {
        return _handle;
      }
    private:
      int _device;
      cudaStream_t _stream;
      cublasHandle_t _handle;
    };

    // We create one cuBLAS/cuDNN handle per host thread. The handle is destroyed
    // when the thread exits.

    static const CudaContext& get_cuda_context() {
      static thread_local const CudaContext cuda_context;
      return cuda_context;
    }

    cudaStream_t get_cuda_stream() {
      return get_cuda_context().cuda_stream();
    }

    cublasHandle_t get_cublas_handle() {
      return get_cuda_context().cublas_handle();
    }

    int get_gpu_count() {
      int gpu_count = 0;
      cudaError_t status = cudaGetDeviceCount(&gpu_count);
      if (status != cudaSuccess)
        return 0;
      return gpu_count;
    }

    bool has_gpu() {
      return get_gpu_count() > 0;
    }

    const cudaDeviceProp& get_device_properties(int device) {
      static thread_local std::vector<std::unique_ptr<cudaDeviceProp>> cache;

      if (device < 0) {
        CUDA_CHECK(cudaGetDevice(&device));
      }
      if (device >= static_cast<int>(cache.size())) {
        cache.resize(device + 1);
      }

      auto& device_prop = cache[device];
      if (!device_prop) {
        device_prop = std::make_unique<cudaDeviceProp>();
        CUDA_CHECK(cudaGetDeviceProperties(device_prop.get(), device));
      }
      return *device_prop;
    }

    // See docs.nvidia.com/deeplearning/sdk/tensorrt-support-matrix/index.html
    // for hardware support of reduced precision.

    bool gpu_supports_int8(int device) {
      const cudaDeviceProp& device_prop = get_device_properties(device);
      return device_prop.major > 6 || (device_prop.major == 6 && device_prop.minor == 1);
    }

    bool gpu_has_int8_tensor_cores(int device) {
      const cudaDeviceProp& device_prop = get_device_properties(device);
      return device_prop.major > 7 || (device_prop.major == 7 && device_prop.minor >= 2);
    }

    bool gpu_has_fp16_tensor_cores(int device) {
      const cudaDeviceProp& device_prop = get_device_properties(device);
      return device_prop.major >= 7;
    }

  }
}
