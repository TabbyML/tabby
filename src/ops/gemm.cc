#include "ctranslate2/ops/gemm.h"

namespace ctranslate2 {
  namespace ops {

    Gemm::Gemm(float alpha, float beta, bool broadcast_c, bool trans_a, bool trans_b)
      : _alpha(alpha)
      , _beta(beta)
      , _broadcast_c(broadcast_c)
      , _trans_a(trans_a)
      , _trans_b(trans_b) {
    }

    void Gemm::operator()(const StorageView& a,
                          const StorageView& b,
                          const StorageView& c,
                          StorageView& y) const {
      switch (a.dtype()) {
      case DataType::DT_INT8:
        DEVICE_DISPATCH(a.device(), (compute<D, int8_t, int32_t>(a, b, c, y)));
        break;
      case DataType::DT_INT16:
        if (a.device() != Device::CPU)
          throw std::invalid_argument("INT16 GEMM is only supported on CPU");
        return compute<Device::CPU, int16_t, int32_t>(a, b, c, y);
      case DataType::DT_FLOAT:
        DEVICE_DISPATCH(a.device(), (compute<D, float>(a, b, c, y)));
        break;
      default:
        throw std::invalid_argument("unsupported compute type " + dtype_name(a.dtype()));
      }
    }

  }
}
