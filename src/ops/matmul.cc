#include "ctranslate2/ops/matmul.h"

#include "device_dispatch.h"

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
      switch (a.dtype()) {
      case DataType::DT_FLOAT:
        DEVICE_DISPATCH(a.device(), (compute<D, float>(a, b, y)));
        break;
      default:
        throw std::invalid_argument("unsupported compute type " + dtype_name(a.dtype()));
      }
    }

  }
}
