#include "ctranslate2/ops/quantize.h"

#include <cmath>

#include "cpu/kernels.h"
#include "cpu/parallel.h"

namespace ctranslate2 {
  namespace ops {

    template<>
    void Quantize::quantize<Device::CPU, float, int8_t>(const StorageView& input,
                                                        StorageView& output,
                                                        StorageView& scale) const {
      // INT8 quantization rescales based on the per batch absolute maximum.
      const dim_t batch_size = scale.size();
      const dim_t depth = input.dim(-1);
      CPU_ISA_DISPATCH(cpu::quantize_s8<ISA>(input.data<float>(),
                                             output.data<int8_t>(),
                                             scale.data<float>(),
                                             batch_size,
                                             depth,
                                             _shift_to_uint8,
                                             _round_before_cast));
    }

    template <typename RoundFunc>
    static void quantize_s16_kernel(const float* x,
                                    const float scale,
                                    int16_t* y,
                                    dim_t size,
                                    const RoundFunc& round_func) {
      constexpr float int16_min = std::numeric_limits<int16_t>::lowest();
      constexpr float int16_max = std::numeric_limits<int16_t>::max();

      cpu::parallel_unary_transform(
        x, y, size, /*work_size=*/5,
        [scale, int16_min, int16_max, &round_func](float v) {
          return std::max(std::min(round_func(v * scale), int16_max), int16_min);
        });
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

      if (_round_before_cast)
        quantize_s16_kernel(input_data, scale_value, output_data, size, std::nearbyintf);
      else
        quantize_s16_kernel(input_data, scale_value, output_data, size, cpu::identity());
    }

  }
}
