#include "ctranslate2/ops/dequantize.h"

#include "../device_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Dequantize::operator()(const StorageView& x, const StorageView& scale, StorageView& y) const {
      PROFILE("Dequantize");
      y.resize_as(x);
      if (x.dtype() == DataType::DT_INT16) {
        if (x.device() != Device::CPU)
          throw std::invalid_argument("INT16 dequantization is only supported on CPU");
        if (!scale.is_scalar())
          throw std::invalid_argument("INT16 quantization scale should be a scalar value");

        primitives<Device::CPU>::dequantize(x.data<int16_t>(),
                                            y.data<float>(),
                                            x.size(),
                                            scale.as_scalar<float>());
      } else if (x.dtype() == DataType::DT_INT8) {
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

    void Dequantize::operator()(const StorageView& gemm_output,
                                const StorageView& input_scale,
                                const StorageView& weight_scale,
                                StorageView& output) const {
      PROFILE("DequantizeGemmOutput");
      output.resize_as(gemm_output);
      if (input_scale.is_scalar() && weight_scale.is_scalar()) {
        if (gemm_output.device() != Device::CPU)
          throw std::invalid_argument("unsupported quantization scales");
        auto scale = input_scale.as_scalar<float>() * weight_scale.as_scalar<float>();
        primitives<Device::CPU>::dequantize(gemm_output.data<int32_t>(),
                                            output.data<float>(),
                                            gemm_output.size(),
                                            scale);
      } else {
        DEVICE_DISPATCH(
          gemm_output.device(),
          primitives<D>::rescale_output(gemm_output.data<int32_t>(),
                                        input_scale.data<float>(),
                                        weight_scale.data<float>(),
                                        output.data<float>(),
                                        input_scale.size(),
                                        gemm_output.dim(-1)));

      }
    }

  }
}
