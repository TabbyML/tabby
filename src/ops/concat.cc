#include "ctranslate2/ops/concat.h"

#include "../device_dispatch.h"

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
      PROFILE_FUN;
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

  }
}
