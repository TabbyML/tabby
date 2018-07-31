#pragma once

#include "op.h"

namespace ctranslate2 {
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
        if (!_split.empty())
          assert(_split.size() == outputs.size());
        size_t axis = _axis < 0 ? input.rank() + _axis : _axis;
        for (size_t j = 0; j < outputs.size(); ++j) {
          auto& x = *outputs[j];
          auto shape = input.shape();
          shape[axis] = _split.empty() ? input.dim(axis) / outputs.size() : _split[j];
          x.resize(shape);
        }
        DEVICE_DISPATCH(input.device(),
                        TYPE_DISPATCH(input.dtype(), (compute<D, T>(input, outputs))));
      }

    private:
      int _axis;
      std::vector<int> _split;

      template <Device D, typename T>
      void compute(const StorageView& input,
                   std::vector<StorageView*>& outputs) const;
    };

  }
}
