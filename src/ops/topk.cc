#include "ctranslate2/ops/topk.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename DataType, typename IndexType>
    void TopK::compute(const StorageView& x,
                       StorageView& values,
                       StorageView& indices) const {
      size_t depth = x.dim(-1);
      size_t batch_size = x.size() / depth;
      for (size_t i = 0; i < batch_size; ++i) {
        const auto* input = x.data<DataType>() + (i * depth);
        auto* val = values.data<DataType>() + (i * _k);
        auto* ind = indices.data<IndexType>() + (i * _k);
        primitives<D>::topk(input, val, ind, _k, depth);
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    TopK::compute<Device::CPU, T, int32_t>(const StorageView& x,        \
                                           StorageView& values,         \
                                           StorageView& indices) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
