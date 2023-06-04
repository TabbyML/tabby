#include "ctranslate2/ops/topk.h"

#include <algorithm>
#include <numeric>

#include "cpu/parallel.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename DataType, typename IndexType>
    void TopK::compute(const StorageView& x,
                       StorageView& values,
                       StorageView& indices) const {
      const dim_t depth = x.dim(-1);
      const dim_t batch_size = x.size() / depth;

      const DataType* x_data = x.data<DataType>();
      DataType* v_data = values.data<DataType>();
      IndexType* i_data = indices.data<IndexType>();

      if (_k == 1) {
        cpu::parallel_for(0, batch_size, 1, [&](dim_t begin, dim_t end) {
          for (dim_t i = begin; i < end; ++i) {
            const DataType* row = x_data + i * depth;
            const DataType* max = std::max_element(row, row + depth);
            v_data[i] = *max;
            i_data[i] = std::distance(row, max);
          }
        });

      } else {
        cpu::parallel_for(0, batch_size, 1, [&](dim_t begin, dim_t end) {
          for (dim_t i = begin; i < end; ++i) {
            const auto* input = x_data + (i * depth);
            auto* val = v_data + (i * _k);
            auto* ind = i_data + (i * _k);

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
        });

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
