#include "ctranslate2/ops/layer_norm.h"

#include "../device_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void LayerNorm::operator()(const StorageView& beta,
                               const StorageView& gamma,
                               const StorageView& input,
                               StorageView& output) const {
      PROFILE("LayerNorm");
      output.resize_as(input);
      DEVICE_DISPATCH(input.device(), (compute<D, float>(beta, gamma, input, output)));
    }

  }
}
