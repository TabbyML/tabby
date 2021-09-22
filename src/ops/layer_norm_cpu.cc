#include "ctranslate2/ops/layer_norm.h"

#include "cpu/kernels.h"

#define EPSILON 1e-5

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void LayerNorm::compute(const StorageView& beta,
                            const StorageView& gamma,
                            const StorageView& input,
                            StorageView& output) const {
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      CPU_ISA_DISPATCH((cpu::layer_norm<ISA>(input.data<T>(),
                                             gamma.data<T>(),
                                             beta.data<T>(),
                                             output.data<T>(),
                                             batch_size,
                                             depth,
                                             static_cast<T>(EPSILON))));
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    LayerNorm::compute<Device::CPU, T>(const StorageView& beta,         \
                                       const StorageView& gamma,        \
                                       const StorageView& input,        \
                                       StorageView& output) const;

    DECLARE_IMPL(float)

  }
}
