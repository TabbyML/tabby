#include "ctranslate2/ops/concat.h"

namespace ctranslate2 {
  namespace ops {

    Concat::Concat(int axis)
      : _axis(axis) {
    }

    void Concat::operator()(const std::vector<StorageView*>& inputs,
                            std::vector<StorageView*>& outputs) const {
      operator()(inputs, *outputs[0]);
    }

    void Concat::operator()(const std::vector<StorageView*>& inputs,
                            StorageView& output) const {
      size_t rank = inputs.front()->rank();
      size_t axis = _axis < 0 ? rank + _axis : _axis;
      size_t concat_dims = 0;
      for (const auto& x : inputs) {
        assert(x->rank() == rank);
        concat_dims += x->dim(axis);
      }

      Shape output_shape(inputs.front()->shape());
      output_shape[axis] = concat_dims;
      output.resize(output_shape);

      DEVICE_DISPATCH(output.device(),
                      TYPE_DISPATCH(output.dtype(), (compute<D, T>(inputs, output))));
    }

    template <Device D, typename T>
    void Concat::compute(const std::vector<StorageView*>& inputs,
                         StorageView& output) const {
      size_t axis = _axis < 0 ? output.rank() + _axis : _axis;
      size_t offset = 0;
      for (const auto& x : inputs) {
        size_t iter_dim = 1;
        size_t copy_dim = 1;
        for (size_t i = 0; i < axis; ++i)
          iter_dim *= x->dim(i);
        for (size_t i = axis; i < x->rank(); ++i)
          copy_dim *= x->dim(i);
        if (copy_dim == 0)
          continue;
        for (size_t i = 0; i < iter_dim; ++i) {
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
