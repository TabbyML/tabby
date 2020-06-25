#include "ctranslate2/ops/softmax.h"

#include "cpu/kernels.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void SoftMax::compute(const StorageView& input,
                          const StorageView* lengths,
                          StorageView& output) const {
      constexpr float epsilon = 0.000001f;
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;

      CPU_ISA_DISPATCH((cpu::softmax<ISA>(input.data<T>(),
                                          lengths ? lengths->data<int32_t>() : nullptr,
                                          output.data<T>(),
                                          lengths ? lengths->dim(0) : 0,
                                          batch_size,
                                          depth,
                                          _log,
                                          epsilon)));
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    SoftMax::compute<Device::CPU, T>(const StorageView& input,          \
                                     const StorageView* lengths,        \
                                     StorageView& output) const;

    DECLARE_IMPL(float)

  }
}
