#include "ctranslate2/ops/layer_norm.h"

#include "cpu/kernels.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void LayerNorm::compute(const StorageView* beta,
                            const StorageView* gamma,
                            const StorageView& input,
                            const dim_t axis,
                            const dim_t outer_size,
                            const dim_t axis_size,
                            const dim_t inner_size,
                            StorageView& output) const {
      if (axis == input.rank() - 1 && beta && gamma) {
        CPU_ISA_DISPATCH((cpu::layer_norm<ISA>(input.data<T>(),
                                               gamma->data<T>(),
                                               beta->data<T>(),
                                               output.data<T>(),
                                               outer_size,
                                               axis_size,
                                               _epsilon)));
      } else {
        CPU_ISA_DISPATCH((cpu::layer_norm_axis<ISA>(input.data<T>(),
                                                    gamma ? gamma->data<T>() : nullptr,
                                                    beta ? beta->data<T>() : nullptr,
                                                    output.data<T>(),
                                                    outer_size,
                                                    axis_size,
                                                    inner_size,
                                                    _epsilon)));
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    LayerNorm::compute<Device::CPU, T>(const StorageView* beta,         \
                                       const StorageView* gamma,        \
                                       const StorageView& input,        \
                                       const dim_t axis,                \
                                       const dim_t outer_size,          \
                                       const dim_t axis_size,           \
                                       const dim_t inner_size,          \
                                       StorageView& output) const;

    DECLARE_IMPL(float)

  }
}
