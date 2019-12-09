#include "ctranslate2/ops/gather.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Gather::compute(const StorageView& data,
                         const StorageView& input,
                         StorageView& output) const {
      const auto* indices = input.data<int32_t>();
      const dim_t copy_dim = data.stride(0);
      for (dim_t i = 0, merge = 1; i < input.size(); i += merge, merge = 1) {
        const dim_t index = indices[i];
        while (i + merge < input.size() && static_cast<dim_t>(indices[i + merge]) == index + merge)
          ++merge;
        const auto* src = data.index<T>({index});
        auto* dst = output.data<T>() + i * copy_dim;
        primitives<Device::CPU>::copy(src, dst, copy_dim * merge);
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
