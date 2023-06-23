#include "ctranslate2/ops/topk.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    TopK::TopK(dim_t k, dim_t axis)
      : _k(k) {
      if (axis != -1)
        throw std::invalid_argument("unsupported topk axis " + std::to_string(axis));
    }

    void TopK::operator()(const StorageView& x, StorageView& values, StorageView& indices) const {
      PROFILE("TopK");
      const dim_t batch_size = x.size() / x.dim(-1);
      values.resize({batch_size, _k});
      indices.resize({batch_size, _k});

      DEVICE_AND_FLOAT_DISPATCH("TopK", x.device(), x.dtype(),
                                (compute<D, T, int32_t>(x, values, indices)));
    }

  }
}
