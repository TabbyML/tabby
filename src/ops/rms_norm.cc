#include "ctranslate2/ops/rms_norm.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    RMSNorm::RMSNorm(const float epsilon)
      : _epsilon(epsilon)
    {
    }

    void RMSNorm::operator()(const StorageView& gamma,
                             const StorageView& input,
                             StorageView& output) const {
      PROFILE("RMSNorm");

      output.resize_as(input);

      DEVICE_AND_FLOAT_DISPATCH("RMSNorm", input.device(), input.dtype(),
                                (compute<D, T>(gamma, input, output)));
    }

  }
}
