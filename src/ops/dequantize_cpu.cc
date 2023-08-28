#include "ctranslate2/ops/dequantize.h"

#include "ctranslate2/ops/gemm.h"

#include "cpu/kernels.h"
#include "cpu/parallel.h"

namespace ctranslate2 {
  namespace ops {

    template <typename T>
    static inline void dequantize_kernel(const T* x,
                                         const float scale,
                                         const dim_t size,
                                         float* y) {
      const float r_scale = 1.f / scale;
      cpu::parallel_unary_transform(x, y, size, /*work_size=*/4,
                                    [r_scale](T v) {
                                      return static_cast<float>(v) * r_scale;
                                    });
    }

    template<>
    void Dequantize::dequantize<Device::CPU, int16_t, float>(const StorageView& input,
                                                             const StorageView& scale,
                                                             StorageView& output) const {
      dequantize_kernel(input.data<int16_t>(),
                        scale.as_scalar<float>(),
                        input.size(),
                        output.data<float>());
    }

    template<>
    void Dequantize::dequantize<Device::CPU, int8_t, float>(const StorageView& input,
                                                            const StorageView& scale,
                                                            StorageView& output) const {
      const dim_t batch_size = scale.size();
      const dim_t depth = input.dim(-1);

      const auto* input_data = input.data<int8_t>();
      const auto* scale_data = scale.data<float>();
      auto* output_data = output.data<float>();

      cpu::parallel_for(0, batch_size, 1, [&](dim_t begin, dim_t end) {
        for (dim_t i = begin; i < end; ++i) {
          const dim_t offset = i * depth;
          dequantize_kernel(input_data + offset, scale_data[i], depth, output_data + offset);
        }
      });
    }

    template<>
    void Dequantize::dequantize_gemm_output<Device::CPU, float>(const StorageView& c,
                                                                const StorageView& a_scale,
                                                                const StorageView& b_scale,
                                                                const bool transpose_a,
                                                                const bool transpose_b,
                                                                const StorageView* bias,
                                                                StorageView& y) const {
      const auto* c_data = c.data<int32_t>();
      auto* y_data = y.data<float>();

      if (a_scale.is_scalar() && b_scale.is_scalar()) {
        const auto scale = a_scale.as_scalar<float>() * b_scale.as_scalar<float>();
        dequantize_kernel(c_data, scale, c.size(), y_data);
        apply_bias_and_activation(y, bias, _activation_type);

      } else {
        const dim_t batch_size = a_scale.size();
        const dim_t depth = c.dim(-1);

        const auto* a_scale_data = a_scale.data<float>();
        const auto* b_scale_data = b_scale.data<float>();

        if (!transpose_a && transpose_b) {
          CPU_ISA_DISPATCH((cpu::dequantize_gemm_output<ISA>(c_data,
                                                             a_scale_data,
                                                             b_scale_data,
                                                             batch_size,
                                                             depth,
                                                             y_data,
                                                             bias ? bias->data<float>() : nullptr,
                                                             _activation_type)));

        } else {
          // Generic implementation.
          cpu::parallel_for(0, batch_size, 1, [&](dim_t begin, dim_t end) {
            for (dim_t i = begin; i < end; ++i) {
              for (dim_t j = 0; j < depth; ++j) {
                const dim_t index = j + i * depth;
                const float scale_a = transpose_a ? a_scale_data[j] : a_scale_data[i];
                const float scale_b = transpose_b ? b_scale_data[j] : b_scale_data[i];
                y_data[index] = static_cast<float>(c_data[index]) / (scale_a * scale_b);
              }
            }
          });

          apply_bias_and_activation(y, bias, _activation_type);
        }
      }
    }

  }
}
