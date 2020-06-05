#include "ctranslate2/ops/dequantize.h"

#include "../device_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Dequantize::operator()(const StorageView& x, const StorageView& scale, StorageView& y) const {
      PROFILE("Dequantize");
      y.resize_as(x);
      if (x.dtype() == DataType::INT16) {
        if (x.device() != Device::CPU)
          throw std::invalid_argument("INT16 dequantization is only supported on CPU");
        if (!scale.is_scalar())
          throw std::invalid_argument("INT16 quantization scale should be a scalar value");

        primitives<Device::CPU>::dequantize(x.data<int16_t>(),
                                            y.data<float>(),
                                            x.size(),
                                            scale.as_scalar<float>());
      } else if (x.dtype() == DataType::INT8) {
        auto batch_size = x.size() / x.dim(-1);
        if (scale.size() != batch_size)
          throw std::invalid_argument("INT8 dequantization expects per-batch scales");

        DEVICE_DISPATCH(
          x.device(),
          primitives<D>::dequantize_batch(x.data<int8_t>(),
                                          scale.data<float>(),
                                          y.data<float>(),
                                          x.size(),
                                          scale.size()));
      } else {
        throw std::invalid_argument("Dequantize: invalid quantized type " + dtype_name(x.dtype())
                                    + ", expected int8 or int16");
      }
    }

    void Dequantize::operator()(const StorageView& c,
                                const StorageView& a_scale,
                                const StorageView& b_scale,
                                const bool transpose_a,
                                const bool transpose_b,
                                StorageView& y) const {
      PROFILE("DequantizeGemmOutput");
      const Device device = c.device();
      y.resize_as(c);
      if (a_scale.is_scalar() && b_scale.is_scalar()) {
        if (device != Device::CPU)
          throw std::invalid_argument("unsupported quantization scales");
        auto scale = a_scale.as_scalar<float>() * b_scale.as_scalar<float>();
        primitives<Device::CPU>::dequantize(c.data<int32_t>(),
                                            y.data<float>(),
                                            c.size(),
                                            scale);
      } else {
        DEVICE_DISPATCH(
          device,
          primitives<D>::rescale_output(c.data<int32_t>(),
                                        a_scale.data<float>(),
                                        b_scale.data<float>(),
                                        transpose_a,
                                        transpose_b,
                                        y.data<float>(),
                                        a_scale.size(),
                                        c.dim(-1)));

      }
    }

  }
}
