#include "ctranslate2/ops/topk.h"

#include "../device_dispatch.h"

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
      PROFILE_FUN;
      size_t batch_size = x.size() / x.dim(-1);
      values.resize({batch_size, _k});
      indices.resize({batch_size, _k});
      DEVICE_DISPATCH(x.device(),
                      (compute<D, float, int32_t>(x, values, indices)));
    }

  }
}
