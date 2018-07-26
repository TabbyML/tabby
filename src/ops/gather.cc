#include "ctranslate2/ops/gather.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Gather::compute(const StorageView& data,
                         const StorageView& input,
                         StorageView& output) const {
      size_t copy_dim = data.stride(0);
      for (size_t i = 0; i < input.size(); ++i) {
        size_t index = input.data<int32_t>()[i];
        const auto* src = data.index<T>({index});
        auto* dst = output.data<T>() + i * copy_dim;
        primitives<Device::CPU>::copy(src, dst, copy_dim);
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
