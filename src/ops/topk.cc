#include "ctranslate2/ops/topk.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "device_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    TopK::TopK(size_t k, int axis)
      : _k(k) {
      if (axis != -1)
        throw std::invalid_argument("unsupported topk axis " + std::to_string(axis));
    }

    void TopK::operator()(const std::vector<StorageView*>& inputs,
                          std::vector<StorageView*>& outputs) const {
      operator()(*inputs[0], *outputs[0], *outputs[1]);
    }

    void TopK::operator()(const StorageView& x, StorageView& values, StorageView& indices) const {
      size_t batch_size = x.size() / x.dim(-1);
      values.resize({batch_size, _k});
      indices.resize({batch_size, _k});
      DEVICE_DISPATCH(x.device(),
                      (compute<D, float, int32_t>(x, values, indices)));
    }

    template <Device D, typename DataType, typename IndexType>
    void TopK::compute(const StorageView& x,
                       StorageView& values,
                       StorageView& indices) const {
      size_t depth = x.dim(-1);
      size_t batch_size = x.size() / depth;
      StorageView full_indices({batch_size, depth}, indices.dtype());

      #pragma omp parallel for
      for (size_t i = 0; i < batch_size; ++i) {
        const auto* input = x.data<DataType>() + (i * depth);
        auto* ids = full_indices.data<IndexType>() + (i * depth);
        auto* val = values.data<DataType>() + (i * _k);
        auto* ind = indices.data<IndexType>() + (i * _k);
        std::iota(ids, ids + depth, 0);
        std::partial_sort(ids, ids + _k, ids + depth,
                          [&input](const IndexType& i1, const IndexType& i2) {
                            return input[i1] > input[i2];
                          });
        for (size_t j = 0; j < _k; ++j) {
          ind[j] = ids[j];
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
