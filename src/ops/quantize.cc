#include "ctranslate2/ops/quantize.h"

#include "../device_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    const StorageView Quantize::default_int16_scale(static_cast<float>(1000));

    Quantize::Quantize(ScaleType int16_scale_type)
      : _int16_scale_type(int16_scale_type) {
      if (int16_scale_type != ScaleType::GLOBAL && int16_scale_type != ScaleType::PER_LAYER)
        throw std::invalid_argument("INT16 quantization only supports GLOBAL and PER_LAYER scales");
    }

    void Quantize::operator()(const std::vector<StorageView*>& inputs,
                              std::vector<StorageView*>& outputs) const {
      operator()(*inputs[0], *outputs[0], *outputs[1]);
    }

    void Quantize::operator()(const StorageView& x,
                              StorageView& y,
                              StorageView& scale,
                              float shift) const {
      PROFILE("Quantize");
      y.resize_as(x);
      if (y.dtype() == DataType::DT_INT16) {
        if (x.device() != Device::CPU)
          throw std::invalid_argument("INT16 quantization is only supported on CPU");
        // INT16 quantization simply rescales by a constant and casts input data.
        if (_int16_scale_type == ScaleType::GLOBAL)
          scale = default_int16_scale;
        else if (_int16_scale_type == ScaleType::PER_LAYER) {
          // The idea is to use 10 bits for the input so that the multiplication is 20
          // bits which gives 12 bits left for accumulation.
          auto max = primitives<Device::CPU>::amax(x.data<float>(), x.size());
          scale = StorageView(static_cast<float>(1 << 10) / max);
        }
        primitives<Device::CPU>::quantize(x.data<float>(),
                                          y.data<int16_t>(),
                                          x.size(),
                                          scale.as_scalar<float>(),
                                          shift);
      } else if (y.dtype() == DataType::DT_INT8) {
        // INT8 quantization rescales based on the per batch absolute maximum.
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
                                        depth,
                                        shift));
      } else {
        throw std::invalid_argument("Quantize: invalid quantized type " + dtype_name(y.dtype())
                                    + ", expected int8 or int16");
      }
    }

  }
}
