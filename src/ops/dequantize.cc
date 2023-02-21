#include "ctranslate2/ops/dequantize.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    Dequantize::Dequantize(const ActivationType* activation_type)
      : _activation_type(activation_type)
    {
    }

    void Dequantize::operator()(const StorageView& input,
                                const StorageView& scale,
                                StorageView& output) const {
      PROFILE("Dequantize");
      output.resize_as(input);

      switch (input.dtype()) {
      case DataType::INT16: {
        if (input.device() != Device::CPU)
          throw std::invalid_argument("INT16 dequantization is only supported on CPU");
        if (!scale.is_scalar())
          throw std::invalid_argument("INT16 quantization scale should be a scalar value");
        dequantize<Device::CPU, int16_t, float>(input, scale, output);
        break;
      }

      case DataType::INT8: {
        const dim_t batch_size = input.size() / input.dim(-1);
        if (scale.size() != batch_size)
          throw std::invalid_argument("INT8 dequantization expects per-batch scales");

        switch (output.dtype()) {
        case DataType::FLOAT32: {
          DEVICE_DISPATCH(input.device(), (dequantize<D, int8_t, float>(input, scale, output)));
          break;
        }

#ifdef CT2_WITH_CUDA
        case DataType::FLOAT16: {
          if (output.device() != Device::CUDA)
            throw std::invalid_argument("Dequantize: float16 ouput is only supported on CUDA");
          dequantize<Device::CUDA, int8_t, float16_t>(input, scale, output);
          break;
        }
#endif

        default:
          throw std::invalid_argument("Dequantize: output should have a float type");
        }

        break;
      }

      default:
        throw std::invalid_argument("Dequantize: invalid quantized type " + dtype_name(input.dtype())
                                    + ", expected int8 or int16");
      }
    }

    void Dequantize::operator()(const StorageView& c,
                                const StorageView& a_scale,
                                const StorageView& b_scale,
                                const bool transpose_a,
                                const bool transpose_b,
                                StorageView& y,
                                const StorageView* bias) const {
      PROFILE("DequantizeGemmOutput");
      y.resize_as(c);

      switch (y.dtype()) {
      case DataType::FLOAT32: {
        DEVICE_DISPATCH(c.device(), (dequantize_gemm_output<D, float>(c,
                                                                      a_scale,
                                                                      b_scale,
                                                                      transpose_a,
                                                                      transpose_b,
                                                                      bias,
                                                                      y)));
        break;
      }

#ifdef CT2_WITH_CUDA
      case DataType::FLOAT16: {
        if (y.device() != Device::CUDA)
          throw std::invalid_argument("DequantizeGemmOutput: float16 ouput is only supported on CUDA");
        dequantize_gemm_output<Device::CUDA, float16_t>(c,
                                                        a_scale,
                                                        b_scale,
                                                        transpose_a,
                                                        transpose_b,
                                                        bias,
                                                        y);
        break;
      }
#endif

      default:
        throw std::invalid_argument("DequantizeGemmOutput: output should have a float type");
      }
    }

  }
}
