#include "ctranslate2/ops/multinomial.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Multinomial::compute(const StorageView& input, StorageView& output) const {
      // TODO: CUDA implementation.
      StorageView input_host(input.shape(), input.dtype());
      StorageView output_host(output.shape(), output.dtype());
      input_host.copy_from(input);
      compute<Device::CPU, T>(input_host, output_host);
      output.copy_from(output_host);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Multinomial::compute<Device::CUDA, T>(const StorageView& input,     \
                                         StorageView& output) const;

    DECLARE_IMPL(float)

  }
}
