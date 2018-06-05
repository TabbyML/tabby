#pragma once

#include "opennmt/ops/op.h"

namespace opennmt {
  namespace ops {

    class Concat : public Op {
    public:
      Concat(int axis)
        : _axis(axis) {
      }

      void operator()(const std::vector<StorageView*>& inputs,
                      std::vector<StorageView*>& outputs) const override {
        operator()(inputs, *outputs[0]);
      }
      void operator()(const std::vector<StorageView*>& inputs,
                      StorageView& output) const {
        TYPE_DISPATCH(output.dtype(), compute<T>(inputs, output));
      }

    private:
      int _axis;

      template <typename T>
      void compute(const std::vector<StorageView*>& inputs,
                   StorageView& output) const {
        size_t rank = inputs.front()->rank();
        size_t axis = _axis < 0 ? rank + _axis : _axis;
        size_t concat_dims = 0;
        for (const auto& x : inputs) {
          concat_dims += x->dim(axis);
        }

        Shape output_shape(inputs.front()->shape());
        output_shape[axis] = concat_dims;
        output.resize(output_shape);

        size_t offset = 0;
        for (const auto& x : inputs) {
          size_t iter_dim = 1;
          size_t copy_dim = 1;
          for (size_t i = 0; i < axis; ++i)
            iter_dim *= x->dim(i);
          for (size_t i = axis; i < x->rank(); ++i)
            copy_dim *= x->dim(i);
          for (size_t i = 0; i < iter_dim; ++i) {
            compute::copy(x->data<T>() + i * copy_dim,
                          output.data<T>() + offset + i * concat_dims * output.stride(axis),
                          copy_dim);
          }
          offset += copy_dim;
        }
      }
    };

  }
}
