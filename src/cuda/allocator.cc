#include "ctranslate2/allocator.h"

#include <memory>

#include "ctranslate2/utils.h"
#include "cuda/utils.h"

#include <cub/util_allocator.cuh>

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

        _allocator.reset(new cub::CachingDeviceAllocator(bin_growth,
                                                         min_bin,
                                                         max_bin,
                                                         max_cached_bytes));
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

  }

  template<>
  Allocator& get_allocator<Device::CUDA>() {
    // Use 1 allocator per thread for performance.
    static thread_local cuda::CubCachingAllocator allocator;
    return allocator;
  }

}
