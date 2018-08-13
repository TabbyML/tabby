#include "ctranslate2/ops/softmax.h"
#include "ctranslate2/ops/log_softmax.h"

#define EPSILON 0.000001f

namespace ctranslate2 {
  namespace ops {

    template <typename T>
    static void softmax(const StorageView& input, StorageView& output, bool log) {
      size_t depth = input.dim(-1);
      size_t batch_size = input.size() / depth;
      for (size_t i = 0; i < batch_size; ++i) {
        const auto* x = input.data<T>() + (i * depth);
        auto* y = output.data<T>() + (i * depth);
        auto max = primitives<>::max(x, depth);
        primitives<>::sub(max, x, y, depth);
        primitives<>::exp(y, y, depth);
        auto sum = primitives<>::sum(y, depth);
        if (log)
          primitives<>::sub(std::log(sum) + max, x, y, depth);
        else
          primitives<>::mul(1.f / (sum + EPSILON), y, depth);
      }
    }

    template <Device D, typename T>
    void SoftMax::compute(const StorageView& input, StorageView& output) const {
      softmax<T>(input, output, false /* log */);
    }

    template <Device D, typename T>
    void LogSoftMax::compute(const StorageView& input, StorageView& output) const {
      softmax<T>(input, output, true /* log */);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    SoftMax::compute<Device::CPU, T>(const StorageView& input,          \
                                     StorageView& output) const;        \
    template void                                                       \
    LogSoftMax::compute<Device::CPU, T>(const StorageView& input,       \
                                        StorageView& output) const;

    DECLARE_IMPL(float)

  }
}
