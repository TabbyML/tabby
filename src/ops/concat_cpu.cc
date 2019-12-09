#include "ctranslate2/ops/concat.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Concat::compute(const std::vector<StorageView*>& inputs,
                         StorageView& output) const {
      const dim_t axis = _axis < 0 ? output.rank() + _axis : _axis;
      dim_t offset = 0;
      for (const auto& x : inputs) {
        dim_t iter_dim = 1;
        dim_t copy_dim = 1;
        for (dim_t i = 0; i < axis; ++i)
          iter_dim *= x->dim(i);
        for (dim_t i = axis; i < x->rank(); ++i)
          copy_dim *= x->dim(i);
        if (copy_dim == 0)
          continue;
        for (dim_t i = 0; i < iter_dim; ++i) {
          primitives<D>::copy(x->data<T>() + i * copy_dim,
                              output.data<T>() + offset + i * output.dim(axis) * output.stride(axis),
                              copy_dim);
        }
        offset += copy_dim;
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Concat::compute<Device::CPU, T>(const std::vector<StorageView*>& inputs, \
                                    StorageView& output) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
