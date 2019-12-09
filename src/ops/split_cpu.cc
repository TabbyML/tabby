#include "ctranslate2/ops/split.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Split::compute(const StorageView& input,
                        std::vector<StorageView*>& outputs) const {
      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      dim_t offset = 0;
      for (auto* output : outputs) {
        auto& x = *output;
        dim_t iter_dim = 1;
        dim_t copy_dim = 1;
        for (dim_t i = 0; i < axis; ++i)
          iter_dim *= x.dim(i);
        for (dim_t i = axis; i < x.rank(); ++i)
          copy_dim *= x.dim(i);
        for (dim_t i = 0; i < iter_dim; ++i) {
          primitives<>::copy(input.data<T>() + offset + i * input.dim(axis) * input.stride(axis),
                             x.data<T>() + i * copy_dim,
                             copy_dim);
        }
        offset += copy_dim;
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Split::compute<Device::CPU, T>(const StorageView& input,            \
                                   std::vector<StorageView*>& outputs) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
