#include "ctranslate2/ops/rms_norm.h"

#include "cpu/kernels.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void RMSNorm::compute(const StorageView& gamma,
                          const StorageView& input,
                          StorageView& output) const {
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      CPU_ISA_DISPATCH((cpu::rms_norm<ISA>(input.data<T>(),
                                           gamma.data<T>(),
                                           output.data<T>(),
                                           batch_size,
                                           depth,
                                           _epsilon,
                                           _use_residual)));
    }

#define DECLARE_IMPL(T)                                                 \
    template void RMSNorm::compute<Device::CPU, T>(const StorageView&,  \
                                                   const StorageView&,  \
                                                   StorageView&) const;

    DECLARE_IMPL(float)

  }
}
