#include "ctranslate2/ops/topk.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename DataType, typename IndexType>
    void TopK::compute(const StorageView& x,
                       StorageView& values,
                       StorageView& indices) const {
      std::vector<IndexType> full_indices;
      size_t depth = x.dim(-1);
      size_t batch_size = x.size() / depth;
      if (depth > full_indices.size())
        full_indices.resize(depth);

      for (size_t i = 0; i < batch_size; ++i) {
        const auto* input = x.data<DataType>() + (i * depth);
        auto* val = values.data<DataType>() + (i * _k);
        auto* ind = indices.data<IndexType>() + (i * _k);
        std::iota(full_indices.begin(), full_indices.begin() + depth, 0);
        std::partial_sort(full_indices.begin(),
                          full_indices.begin() + _k,
                          full_indices.begin() + depth,
                          [&input](const IndexType& i1, const IndexType& i2) {
                            return input[i1] > input[i2];
                          });
        for (size_t j = 0; j < _k; ++j) {
          ind[j] = full_indices[j];
          val[j] = input[ind[j]];
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
