#include "ctranslate2/ops/topk.h"

#include <algorithm>
#include <numeric>

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename DataType, typename IndexType>
    void TopK::compute(const StorageView& x,
                       StorageView& values,
                       StorageView& indices) const {
      const dim_t depth = x.dim(-1);
      const dim_t batch_size = x.size() / depth;

      #pragma omp parallel for
      for (dim_t i = 0; i < batch_size; ++i) {
        const auto* input = x.data<DataType>() + (i * depth);
        auto* val = values.data<DataType>() + (i * _k);
        auto* ind = indices.data<IndexType>() + (i * _k);

        if (_k == 1) {
          const DataType* max = std::max_element(input, input + depth);
          const IndexType index = std::distance(input, max);
          val[0] = *max;
          ind[0] = index;
        } else {
          StorageView range({depth}, indices.dtype());
          auto* ids = range.data<IndexType>();
          std::iota(ids, ids + depth, 0);
          std::partial_sort(ids, ids + _k, ids + depth,
                            [&input](const IndexType& i1, const IndexType& i2) {
                              return input[i1] > input[i2];
                            });
          for (dim_t j = 0; j < _k; ++j) {
            ind[j] = ids[j];
            val[j] = input[ind[j]];
          }
        }
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
