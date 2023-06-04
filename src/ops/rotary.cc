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
                            StorageView& output) const {
      output.resize_as(input);

      switch (input.dtype()) {
      case DataType::FLOAT32: {
        DEVICE_DISPATCH(input.device(), (compute<D, float>(input, sin, cos, output)));
        break;
      }
#ifdef CT2_WITH_CUDA
      case DataType::FLOAT16: {
        if (input.device() != Device::CUDA)
          throw std::invalid_argument("FP16 Rotary is only supported on GPU");
        compute<Device::CUDA, float16_t>(input, sin, cos, output);
        break;
      }
#endif
      default:
        throw std::invalid_argument("Rotary only supports float types");
      }

    }

  }
}
