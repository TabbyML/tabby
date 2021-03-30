#include "ctranslate2/ops/matmul.h"

#include "device_dispatch.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    MatMul::MatMul(bool trans_a, bool trans_b, float alpha)
      : _trans_a(trans_a)
      , _trans_b(trans_b)
      , _alpha(alpha) {
    }

    void MatMul::operator()(const StorageView& a,
                            const StorageView& b,
                            StorageView& y) const {
      PROFILE("MatMul");
      switch (a.dtype()) {
      case DataType::FLOAT:
        DEVICE_DISPATCH(a.device(), (compute<D, float>(a, b, y)));
        break;
#ifdef CT2_WITH_CUDA
      case DataType::FLOAT16:
        if (a.device() != Device::CUDA)
          throw std::invalid_argument("FP16 MatMul is only supported on CUDA");
        compute<Device::CUDA, float16_t>(a, b, y);
        break;
#endif
      default:
        throw std::invalid_argument("MatMul: unsupported compute type " + dtype_name(a.dtype()));
      }
    }

  }
}
