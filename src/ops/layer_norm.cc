#include "ctranslate2/ops/layer_norm.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void LayerNorm::operator()(const StorageView& beta,
                               const StorageView& gamma,
                               const StorageView& input,
                               StorageView& output) const {
      PROFILE("LayerNorm");
      output.resize_as(input);
      switch (input.dtype()) {
      case DataType::FLOAT32: {
        DEVICE_DISPATCH(input.device(), (compute<D, float>(beta, gamma, input, output)));
        break;
      }
#ifdef CT2_WITH_CUDA
      case DataType::FLOAT16: {
        if (input.device() != Device::CUDA)
          throw std::invalid_argument("FP16 LayerNorm is only supported on GPU");
        compute<Device::CUDA, float16_t>(beta, gamma, input, output);
        break;
      }
#endif
      default:
        throw std::invalid_argument("LayerNorm only supports float (or float16 on GPU)");
      }
    }

  }
}
