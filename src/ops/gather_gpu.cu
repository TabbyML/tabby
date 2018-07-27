#include "ctranslate2/ops/gather.h"

#include <thrust/gather.h>

#include "ctranslate2/cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Gather::compute(const StorageView& data,
                         const StorageView& input,
                         StorageView& output) const {
      static thread_local StorageView map(input.dtype(), Device::CPU);
      static thread_local StorageView map_device(input.dtype(), Device::CUDA);
      map.resize({output.size()});
      map_device.resize({output.size()});

      auto* map_raw = map.data<int32_t>();
      auto* map_device_raw = map_device.data<int32_t>();

      size_t copy_dim = data.stride(0);
      size_t map_size = map.size();

      for (size_t i = 0; i < input.size(); ++i) {
        size_t index = input.data<int32_t>()[i];
        size_t map_offset = i * copy_dim;
        std::iota(map_raw + map_offset, map_raw + map_offset + copy_dim, index * copy_dim);
      }

      cross_device_primitives<Device::CPU, Device::CUDA>::copy(map_raw, map_device_raw, map_size);
      thrust::gather(thrust::cuda::par.on(cuda::get_cuda_stream()),
                     map_device_raw, map_device_raw + map_size, data.data<T>(), output.data<T>());
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Gather::compute<Device::CUDA, T>(const StorageView& data,           \
                                     const StorageView& input,          \
                                     StorageView& output) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
