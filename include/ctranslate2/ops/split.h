#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Split : public Op {
    public:
      Split(int axis, bool no_copy = false)
        : _axis(axis)
        , _no_copy(no_copy) {
        check_arguments();
      }
      Split(int axis, const std::vector<int>& split, bool no_copy = false)
        : _axis(axis)
        , _split(split)
        , _no_copy(no_copy) {
        check_arguments();
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
        size_t offset = 0;
        for (size_t j = 0; j < outputs.size(); ++j) {
          auto& x = *outputs[j];
          auto shape = input.shape();
          auto split_size = _split.empty() ? input.dim(axis) / outputs.size() : _split[j];
          shape[axis] = split_size;
          if (_no_copy) {
            TYPE_DISPATCH(input.dtype(),
                          x.view(const_cast<T*>(input.data<T>() + offset), shape));
          } else {
            x.resize(shape);
          }
          offset += input.stride(0) * split_size;
        }

        if (!_no_copy) {
          DEVICE_DISPATCH(input.device(),
                          TYPE_DISPATCH(input.dtype(), (compute<D, T>(input, outputs))));
        }
      }

    private:
      int _axis;
      std::vector<int> _split;
      bool _no_copy;

      void check_arguments() const {
        if (_no_copy && _axis != 0)
          throw std::invalid_argument("no_copy is only defined when splitting across the first dimension");
      }

      template <Device D, typename T>
      void compute(const StorageView& input,
                   std::vector<StorageView*>& outputs) const;
    };

  }
}
