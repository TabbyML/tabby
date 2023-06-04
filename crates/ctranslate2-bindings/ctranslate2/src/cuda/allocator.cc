#include "ctranslate2/allocator.h"

#include <memory>
#include <mutex>

#include "ctranslate2/utils.h"
#include "cuda/utils.h"
#include "env.h"

#include <cuda.h>
#include <cub/util_allocator.cuh>
#include <spdlog/spdlog.h>

namespace ctranslate2 {
  namespace cuda {

    // See https://nvlabs.github.io/cub/structcub_1_1_caching_device_allocator.html.
    class CubCachingAllocator : public Allocator {
    public:
      CubCachingAllocator() {
        unsigned int bin_growth = 4;
        unsigned int min_bin = 3;
        unsigned int max_bin = 12;
        size_t max_cached_bytes = 200 * (1 << 20);  // 200MB

        const char* config_env = std::getenv("CT2_CUDA_CACHING_ALLOCATOR_CONFIG");
        if (config_env) {
          const std::vector<std::string> values = split_string(config_env, ',');
          if (values.size() != 4)
            throw std::invalid_argument("CT2_CUDA_CACHING_ALLOCATOR_CONFIG environment variable "
                                        "should have format: "
                                        "bin_growth,min_bin,max_bin,max_cached_bytes");
          bin_growth = std::stoul(values[0]);
          min_bin = std::stoul(values[1]);
          max_bin = std::stoul(values[2]);
          max_cached_bytes = std::stoull(values[3]);
        }

        _allocator = std::make_unique<cub::CachingDeviceAllocator>(bin_growth,
                                                                   min_bin,
                                                                   max_bin,
                                                                   max_cached_bytes);
      }

      void* allocate(size_t size, int device_index) override {
        void* ptr = nullptr;
        CUDA_CHECK(_allocator->DeviceAllocate(device_index, &ptr, size, cuda::get_cuda_stream()));
        return ptr;
      }

      void free(void* ptr, int device_index) override {
        _allocator->DeviceFree(device_index, ptr);
      }

      void clear_cache() override {
        _allocator->FreeAllCached();
      }

    private:
      std::unique_ptr<cub::CachingDeviceAllocator> _allocator;
    };

    class CudaAsyncAllocator : public Allocator {
    public:
      void* allocate(size_t size, int device_index) override {
#if CUDA_VERSION >= 11020
        int prev_device_index = -1;
        if (device_index >= 0) {
          CUDA_CHECK(cudaGetDevice(&prev_device_index));
          CUDA_CHECK(cudaSetDevice(device_index));
        }

        void* ptr = nullptr;
        CUDA_CHECK(cudaMallocAsync(&ptr, size, get_cuda_stream()));

        if (prev_device_index >= 0) {
          CUDA_CHECK(cudaSetDevice(prev_device_index));
        }

        return ptr;
#else
        (void)size;
        (void)device_index;
        throw std::runtime_error("The asynchronous CUDA allocator requires CUDA >= 11.2");
#endif
      }

      void free(void* ptr, int device_index) override {
#if CUDA_VERSION >= 11020
        int prev_device_index = -1;
        if (device_index >= 0) {
          CUDA_CHECK(cudaGetDevice(&prev_device_index));
          CUDA_CHECK(cudaSetDevice(device_index));
        }

        CUDA_CHECK(cudaFreeAsync(ptr, get_cuda_stream()));

        if (prev_device_index >= 0) {
          CUDA_CHECK(cudaSetDevice(prev_device_index));
        }
#else
        (void)ptr;
        (void)device_index;
        throw std::runtime_error("The asynchronous CUDA allocator requires CUDA >= 11.2");
#endif
      }
    };

    static bool support_cuda_malloc_async() {
#if CUDA_VERSION < 11020
      return false;
#else
      for (int i = 0; i < get_gpu_count(); ++i) {
        int supported = 0;
        cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, i);
        if (!supported)
          return false;
      }
      return true;
#endif
    }

    enum class CudaAllocator {
      CubCaching,
      MallocAsync,
    };

    static CudaAllocator resolve_cuda_allocator() {
      const bool cuda_malloc_async_is_supported = support_cuda_malloc_async();
      const auto allocator_name = read_string_from_env("CT2_CUDA_ALLOCATOR",
                                                       cuda_malloc_async_is_supported
                                                       ? "cuda_malloc_async"
                                                       : "cub_caching");

      CudaAllocator allocator = CudaAllocator::MallocAsync;

      if (allocator_name == "cub_caching") {
        allocator = CudaAllocator::CubCaching;
      } else if (allocator_name == "cuda_malloc_async") {
        if (!cuda_malloc_async_is_supported)
          throw std::runtime_error("The asynchronous CUDA allocator requires CUDA >= 11.2");
        allocator = CudaAllocator::MallocAsync;
      } else {
        throw std::invalid_argument("Invalid CUDA allocator " + allocator_name);
      }

      static std::once_flag log_once_flag;
      std::call_once(log_once_flag, [&allocator_name]() {
        spdlog::info("Using CUDA allocator: {}", allocator_name);
      });

      return allocator;
    }

  }

  template<>
  Allocator& get_allocator<Device::CUDA>() {
    static const cuda::CudaAllocator cuda_allocator = cuda::resolve_cuda_allocator();

    if (cuda_allocator == cuda::CudaAllocator::CubCaching) {
      // Use 1 allocator per thread for performance.
      static thread_local cuda::CubCachingAllocator allocator;
      return allocator;
    }

    static cuda::CudaAsyncAllocator allocator;
    return allocator;
  }

}
