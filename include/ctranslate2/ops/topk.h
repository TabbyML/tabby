#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class TopK : public Op {
    public:
      TopK(size_t k, int axis = -1)
        : _k(k)
        , _axis(axis) {
        if (axis != -1)
          throw std::invalid_argument("unsupported topk axis " + std::to_string(axis));
      }

      void operator()(const std::vector<StorageView*>& inputs,
                      std::vector<StorageView*>& outputs) const override {
        operator()(*inputs[0], *outputs[0], *outputs[1]);
      }

      void operator()(const StorageView& x, StorageView& values, StorageView& indices) const {
        compute<float, int32_t>(x, values, indices);
      }

    private:
      size_t _k;
      int _axis;

      template <typename DataType, typename IndexType>
      void compute(const StorageView& x,
                   StorageView& values,
                   StorageView& indices) const {
        size_t depth = x.dim(-1);
        size_t batch_size = x.size() / depth;
        StorageView tmp({depth}, indices.dtype());
        values.resize({batch_size, _k});
        indices.resize({batch_size, _k});
        for (size_t i = 0; i < batch_size; ++i) {
          const auto* input = x.data<DataType>() + (i * depth);
          primitives::topk(input, tmp.data<IndexType>(), _k, depth);
          auto* val = values.data<DataType>() + (i * _k);
          auto* ind = indices.data<IndexType>() + (i * _k);
          primitives::copy(tmp.data<IndexType>(), ind, _k);
          for (size_t j = 0; j < _k; ++j)
            val[j] = input[ind[j]];
        }
      }

    };

  }
}
