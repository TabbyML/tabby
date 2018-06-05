#pragma once

#include "opennmt/ops/op.h"

namespace opennmt {
  namespace ops {

    class Split : public Op {
    public:
      Split(int axis)
        : _axis(axis) {
      }
      Split(int axis, const std::vector<int>& split)
        : _axis(axis)
        , _split(split) {
      }

      void operator()(const std::vector<StorageView*>& inputs,
                      std::vector<StorageView*>& outputs) const override {
        operator()(*inputs[0], outputs);
      }
      void operator()(const StorageView& input, StorageView& output1, StorageView& output2) const {
        std::vector<StorageView*> outputs{&output1, &output2};
        operator()(input, outputs);
      }
      void operator()(const StorageView& input,
                      StorageView& output1, StorageView& output2, StorageView& output3) const {
        std::vector<StorageView*> outputs{&output1, &output2, &output3};
        operator()(input, outputs);
      }
      void operator()(const StorageView& input,
                      std::vector<StorageView*>& outputs) const {
        TYPE_DISPATCH(input.dtype(), compute<T>(input, outputs));
      }

    private:
      int _axis;
      std::vector<int> _split;

      template <typename T>
      void compute(const StorageView& input,
                   std::vector<StorageView*>& outputs) const {
        size_t rank = input.rank();
        size_t axis = _axis < 0 ? rank + _axis : _axis;
        size_t offset = 0;
        if (!_split.empty())
          assert(_split.size() == outputs.size());
        for (size_t j = 0; j < outputs.size(); ++j) {
          auto& x = *outputs[j];
          auto shape = input.shape();
          shape[axis] = _split.empty() ? input.dim(axis) / outputs.size() : _split[j];
          x.resize(shape);
          size_t iter_dim = 1;
          size_t copy_dim = 1;
          for (size_t i = 0; i < axis; ++i)
            iter_dim *= x.dim(i);
          for (size_t i = axis; i < x.rank(); ++i)
            copy_dim *= x.dim(i);
          for (size_t i = 0; i < iter_dim; ++i) {
            primitives::copy(input.data<T>() + offset + i * input.dim(axis) * input.stride(axis),
                             x.data<T>() + i * copy_dim,
                             copy_dim);
          }
          offset += copy_dim;
        }
      }

    };

  }
}
