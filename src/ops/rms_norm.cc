#include "ctranslate2/ops/rms_norm.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void RMSNorm::operator()(const StorageView& gamma,
                             const StorageView& input,
                             StorageView& output) const {
      PROFILE("RMSNorm");

      output.resize_as(input);

      switch (input.dtype()) {
      case DataType::FLOAT32: {
        DEVICE_DISPATCH(input.device(), (compute<D, float>(gamma, input, output)));
        break;
      }
#ifdef CT2_WITH_CUDA
      case DataType::FLOAT16: {
        if (input.device() != Device::CUDA)
          throw std::invalid_argument("FP16 RMSNorm is only supported on GPU");
        compute<Device::CUDA, float16_t>(gamma, input, output);
        break;
      }
#endif
      default:
        throw std::invalid_argument("RMSNorm only supports float types");
      }
    }

  }
}
