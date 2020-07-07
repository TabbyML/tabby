#include "ctranslate2/ops/gather.h"

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Gather::compute(const StorageView& data,
                         const StorageView& input,
                         StorageView& output) const {
      const auto* indices = input.data<int32_t>();
      const dim_t copy_size = data.stride(0);
      const dim_t num_indices = input.size();

      const T* src = data.data<T>();
      T* dst = output.data<T>();

      #pragma omp parallel for
      for (dim_t i = 0; i < num_indices; ++i) {
        const int32_t read_index = indices[i];
        primitives<Device::CPU>::copy(src + read_index * copy_size, dst + i * copy_size, copy_size);
      }
    }

#define DECLARE_IMPL(T)                                         \
    template void                                               \
    Gather::compute<Device::CPU, T>(const StorageView& data,    \
                                    const StorageView& input,   \
                                    StorageView& output) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
