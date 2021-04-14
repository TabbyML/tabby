#include "ctranslate2/ops/multinomial.h"

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Multinomial::compute(const StorageView& input, StorageView& output) const {
      // TODO: CUDA implementation.
      StorageView output_host(output.shape(), output.dtype());
      dispatch(input.to(Device::CPU), output_host);
      output.copy_from(output_host);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Multinomial::compute<Device::CUDA, T>(const StorageView& input,     \
                                          StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)

  }
}
