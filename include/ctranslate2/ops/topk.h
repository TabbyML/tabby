#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class TopK : public Op {
    public:
      TopK(size_t k, int axis = -1)
        : _k(k) {
        if (axis != -1)
          throw std::invalid_argument("unsupported topk axis " + std::to_string(axis));
      }

      void operator()(const std::vector<StorageView*>& inputs,
                      std::vector<StorageView*>& outputs) const override {
        operator()(*inputs[0], *outputs[0], *outputs[1]);
      }

      void operator()(const StorageView& x, StorageView& values, StorageView& indices) const {
        size_t batch_size = x.size() / x.dim(-1);
        values.resize({batch_size, _k});
        indices.resize({batch_size, _k});
        DEVICE_DISPATCH(x.device(),
                        (compute<D, float, int32_t>(x, values, indices)));
      }

    private:
      size_t _k;

      template <Device D, typename DataType, typename IndexType>
      void compute(const StorageView& x,
                   StorageView& values,
                   StorageView& indices) const;

    };

  }
}
