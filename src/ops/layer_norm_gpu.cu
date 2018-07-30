#include "ctranslate2/ops/layer_norm.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void LayerNorm::compute(const StorageView& beta,
                            const StorageView& gamma,
                            const StorageView& input,
                            StorageView& output) const {
      // TODO: GPU impl.

      static thread_local StorageView beta_host;
      static thread_local StorageView gamma_host;
      static thread_local StorageView input_host;
      static thread_local StorageView output_host;

      beta_host = beta;
      gamma_host = gamma;
      input_host = input;

      compute<Device::CPU, T>(beta_host, gamma_host, input_host, output_host);

      output = output_host;
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    LayerNorm::compute<Device::CUDA, T>(const StorageView& beta,        \
                                        const StorageView& gamma,       \
                                        const StorageView& input,       \
                                        StorageView& output) const;

    DECLARE_IMPL(float)

  }
}
