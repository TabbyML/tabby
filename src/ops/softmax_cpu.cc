#include "ctranslate2/ops/softmax.h"

#include <cmath>

#define EPSILON 0.000001f

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void SoftMax::compute(const StorageView& input,
                          const StorageView* lengths,
                          StorageView& output) const {
      size_t total_depth = input.dim(-1);
      size_t batch_size = input.size() / total_depth;
      #pragma omp parallel for
      for (long long i = 0; i < static_cast<long long>(batch_size); ++i) {
        const auto* x = input.data<T>() + (i * total_depth);
        auto* y = output.data<T>() + (i * total_depth);
        size_t depth = total_depth;
        if (lengths) {
          // Directly set 0 in output for out of range positions.
          size_t batch_index = i * lengths->dim(0) / batch_size;
          depth = lengths->at<int32_t>(batch_index);
          primitives<>::fill(y + depth, static_cast<float>(0), total_depth - depth);
        }
        auto max = primitives<>::max(x, depth);
        primitives<>::sub(max, x, y, depth);
        primitives<>::exp(y, y, depth);
        auto sum = primitives<>::sum(y, depth);
        if (_log)
          primitives<>::sub(std::log(sum) + max, x, y, depth);
        else
          primitives<>::mul(1.f / (sum + EPSILON), y, depth);
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    SoftMax::compute<Device::CPU, T>(const StorageView& input,          \
                                     const StorageView* lengths,        \
                                     StorageView& output) const;

    DECLARE_IMPL(float)

  }
}
