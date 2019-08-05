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

    // INT8 quantization rescales based on the per batch absolute maximum.
    class QuantizeINT8 : public Op {
    public:
      void operator()(const std::vector<StorageView*>& inputs,
                      std::vector<StorageView*>& outputs) const override {
        operator()(*inputs[0], *outputs[0], *outputs[1]);
      }
      void operator()(const StorageView& x, StorageView& y, StorageView& scale) const {
        auto depth = x.dim(-1);
        auto batch_size = x.size() / depth;
        y.resize_as(x);
        scale.resize({batch_size});
        DEVICE_DISPATCH(
          x.device(),
          primitives<D>::quantize_batch(x.data<float>(),
                                        scale.data<float>(),
                                        y.data<int8_t>(),
                                        batch_size,
                                        depth));
      }
    };

  }
}
