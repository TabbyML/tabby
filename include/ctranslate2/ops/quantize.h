#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    // INT16 quantization simply rescales by a constant and casts input data.
    class QuantizeINT16 : public BinaryOp {
    public:
      void operator()(const StorageView& x, const StorageView& scale, StorageView& y) const override {
        if (x.device() != Device::CPU)
          throw std::invalid_argument("INT16 quantization is only supported on CPU");
        if (!scale.is_scalar())
          throw std::invalid_argument("INT16 quantization scale should be a scalar value");

        y.resize_as(x);
        primitives<Device::CPU>::quantize(x.data<float>(),
                                          y.data<int16_t>(),
                                          x.size(),
                                          scale.as_scalar<float>());
      }
    };

  }
}
