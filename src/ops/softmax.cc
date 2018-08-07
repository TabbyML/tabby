#include "ctranslate2/ops/softmax.h"

#define EPSILON 0.000001f

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void SoftMax::compute(const StorageView& input, StorageView& output) const {
      size_t depth = input.dim(-1);
      size_t batch_size = input.size() / depth;
      for (size_t i = 0; i < batch_size; ++i) {
        const auto* x = input.data<T>() + (i * depth);
        auto* y = output.data<T>() + (i * depth);
        auto max = primitives<>::max(x, depth);
        primitives<>::sub(max, x, y, depth);
        primitives<>::exp(y, y, depth);
        auto sum = primitives<>::sum(y, depth);
        primitives<>::mul(1.f / (sum + EPSILON), y, depth);
      }
    }

#define DECLARE_IMPL(T)                                         \
    template void                                               \
    SoftMax::compute<Device::CPU, T>(const StorageView& input,  \
                                     StorageView& output) const;

    DECLARE_IMPL(float)

  }
}
