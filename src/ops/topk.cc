#include "ctranslate2/ops/topk.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    TopK::TopK(dim_t k, dim_t axis)
      : _k(k) {
      if (axis != -1)
        throw std::invalid_argument("unsupported topk axis " + std::to_string(axis));
    }

    void TopK::operator()(const StorageView& x, StorageView& values, StorageView& indices) const {
      PROFILE("TopK");
      const dim_t batch_size = x.size() / x.dim(-1);
      values.resize({batch_size, _k});
      indices.resize({batch_size, _k});

      switch (x.dtype()) {
      case DataType::FLOAT32: {
        DEVICE_DISPATCH(x.device(),
                        (compute<D, float, int32_t>(x, values, indices)));
        break;
      }
#ifdef CT2_WITH_CUDA
      case DataType::FLOAT16: {
        if (x.device() != Device::CUDA)
          throw std::invalid_argument("FP16 TopK is only supported on GPU");
        compute<Device::CUDA, float16_t, int32_t>(x, values, indices);
        break;
      }
#endif
      default:
        throw std::invalid_argument("TopK only supports float (or float16 on GPU)");
      }
    }

  }
}
