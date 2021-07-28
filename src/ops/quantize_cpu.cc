#include "ctranslate2/ops/quantize.h"

#include "cpu/kernels.h"
#include "cpu/parallel.h"

namespace ctranslate2 {
  namespace ops {

    template<>
    void Quantize::quantize<Device::CPU, float, int8_t>(const StorageView& input,
                                                        StorageView& output,
                                                        StorageView& scale) const {
      // INT8 quantization rescales based on the per batch absolute maximum.
      constexpr float int8_min = std::numeric_limits<int8_t>::min();

      const dim_t batch_size = scale.size();
      const dim_t depth = input.dim(-1);
      const float shift = (_shift_to_uint8 ? -int8_min : 0);

      CPU_ISA_DISPATCH(cpu::quantize_s8<ISA>(input.data<float>(),
                                             output.data<int8_t>(),
                                             scale.data<float>(),
                                             batch_size,
                                             depth,
                                             shift));
    }

    template<>
    void Quantize::quantize<Device::CPU, float, int16_t>(const StorageView& input,
                                                         StorageView& output,
                                                         StorageView& scale) const {
      // INT16 quantization simply rescales by a constant.

      const dim_t size = input.size();
      const auto* input_data = input.data<float>();
      auto* output_data = output.data<int16_t>();

      float scale_value = global_int16_scale;
      if (_int16_scale_type == ScaleType::PER_LAYER) {
        // The idea is to use 10 bits for the input so that the multiplication is 20
        // bits which gives 12 bits left for accumulation.
        const float amax = primitives<Device::CPU>::amax(input_data, size);
        scale_value = static_cast<float>(1 << 10) / amax;
      }

      scale = StorageView(scale_value);

      cpu::parallel_unary_transform(
        input_data, output_data, size, /*work_size=*/5,
        [scale_value](float v) {
          return static_cast<int16_t>(
            std::max(
              std::min(v * scale_value,
                       static_cast<float>(std::numeric_limits<int16_t>::max())),
              static_cast<float>(std::numeric_limits<int16_t>::lowest())));
        });
    }

  }
}
