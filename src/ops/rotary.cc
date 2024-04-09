#include "ctranslate2/ops/rotary.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    Rotary::Rotary(const dim_t ndims, const bool interleave)
      : _ndims(ndims)
      , _interleave(interleave)
    {
    }

    void Rotary::operator()(const StorageView& input,
                            const StorageView& sin,
                            const StorageView& cos,
                            StorageView& output,
                            bool is_transposed) const {
      PROFILE("Rotary");

      output.resize_as(input);

      DEVICE_AND_FLOAT_DISPATCH("Rotary", input.device(), input.dtype(),
                                (compute<D, T>(input, sin, cos, output, is_transposed)));
    }

  }
}
