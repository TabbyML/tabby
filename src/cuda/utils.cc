#include "./utils.h"

#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ctranslate2/primitives/primitives.h"
#include "ctranslate2/utils.h"

namespace ctranslate2 {
  namespace cuda {

    std::string cublasGetStatusString(cublasStatus_t status)
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

    cudaStream_t get_cuda_stream() {
      // Only use the default stream for now.
      return static_cast<cudaStream_t>(0);
    }

    class CublasHandle {
    public:
      CublasHandle() {
        CUDA_CHECK(cudaGetDevice(&_device));
        CUBLAS_CHECK(cublasCreate(&_handle));
        CUBLAS_CHECK(cublasSetStream(_handle, get_cuda_stream()));
      }
      ~CublasHandle() {
        ScopedDeviceSetter scoped_device_setter(Device::CUDA, _device);
        cublasDestroy(_handle);
      }
      cublasHandle_t get() const {
        return _handle;
      }
    private:
      int _device;
      cublasHandle_t _handle;
    };

    // We create one cuBLAS/cuDNN handle per host thread. The handle is destroyed
    // when the thread exits.

    cublasHandle_t get_cublas_handle() {
      static thread_local CublasHandle cublas_handle;
      return cublas_handle.get();
    }

    CachingAllocatorConfig get_caching_allocator_config() {
      CachingAllocatorConfig config;
      const char* config_env = std::getenv("CT2_CUDA_CACHING_ALLOCATOR_CONFIG");
      if (config_env) {
        const std::vector<std::string> values = split_string(config_env, ',');
        if (values.size() != 4)
          throw std::invalid_argument("CT2_CUDA_CACHING_ALLOCATOR_CONFIG environment variable "
                                      "should have format: "
                                      "bin_growth,min_bin,max_bin,max_cached_bytes");
        config.bin_growth = std::stoul(values[0]);
        config.min_bin = std::stoul(values[1]);
        config.max_bin = std::stoul(values[2]);
        config.max_cached_bytes = std::stoull(values[3]);
      }
      return config;
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

    static const cudaDeviceProp& get_device_properties(int device) {
      static std::unordered_map<int, cudaDeviceProp> cache;
      static std::mutex mutex;

      if (device < 0) {
        CUDA_CHECK(cudaGetDevice(&device));
      }

      const std::lock_guard<std::mutex> lock(mutex);
      auto it = cache.find(device);
      if (it != cache.end()) {
        return it->second;
      }

      cudaDeviceProp device_prop;
      CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device));
      return cache.emplace(device, device_prop).first->second;
    }

    // See docs.nvidia.com/deeplearning/sdk/tensorrt-support-matrix/index.html
    // for hardware support of reduced precision.

    bool has_fast_int8(int device) {
      const cudaDeviceProp& device_prop = get_device_properties(device);
      return device_prop.major > 6 || (device_prop.major == 6 && device_prop.minor == 1);
    }

    bool has_fast_float16(int device) {
      const cudaDeviceProp& device_prop = get_device_properties(device);
      return device_prop.major >= 7;
    }

    ThrustAllocator::value_type* ThrustAllocator::allocate(std::ptrdiff_t num_bytes) {
      return reinterpret_cast<ThrustAllocator::value_type*>(
        primitives<Device::CUDA>::alloc_data(num_bytes));
    }

    void ThrustAllocator::deallocate(ThrustAllocator::value_type* p, size_t) {
      return primitives<Device::CUDA>::free_data(p);
    }

    ThrustAllocator& get_thrust_allocator() {
      static ThrustAllocator thrust_allocator;
      return thrust_allocator;
    }

  }
}
