#include "ctranslate2/ops/topp_mask.h"

#include "ctranslate2/ops/softmax.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    TopPMask::TopPMask(const float p, const float mask_value)
      : _p(p)
      , _mask_value(mask_value)
    {
    }

    void TopPMask::operator()(const StorageView& input, StorageView& output) const {
      PROFILE("TopPMask");

      const DataType dtype = input.dtype();
      const Device device = input.device();

      StorageView probs(dtype, device);
      ops::SoftMax()(input, probs);

      output.resize_as(input);

      switch (dtype) {
      case DataType::FLOAT32: {
        DEVICE_DISPATCH(device, (compute<D, float>(input, probs, output)));
        break;
      }
#ifdef CT2_WITH_CUDA
      case DataType::FLOAT16: {
        if (device != Device::CUDA)
          throw std::invalid_argument("FP16 TopPMask is only supported on GPU");
        compute<Device::CUDA, float16_t>(input, probs, output);
        break;
      }
#endif
      default:
        throw std::invalid_argument("TopPMask only supports float types");
      }
    }

    dim_t TopPMask::max_num_classes(const Device device) {
      dim_t num_classes = 0;
      DEVICE_DISPATCH(device, num_classes = max_num_classes<D>());
      return num_classes;
    }

  }
}
