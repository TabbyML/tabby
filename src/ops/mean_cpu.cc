#include "ctranslate2/ops/mean.h"

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Mean::compute(const StorageView& input,
                       const dim_t outer_size,
                       const dim_t axis_size,
                       const dim_t inner_size,
                       StorageView& output) const {
      const auto* src = input.data<T>();
      auto* dst = output.data<T>();

      #pragma omp parallel for
      for (dim_t i = 0; i < outer_size; ++i) {
        for (dim_t j = 0; j < inner_size; ++j) {
          float sum = 0.f;
          for (dim_t k = 0; k < axis_size; ++k) {
            sum += src[i * axis_size * inner_size + k * inner_size + j];
          }
          dst[i * inner_size + j] = sum / float(axis_size);
        }
      }
    }

#define DECLARE_IMPL(T)                                         \
    template void                                               \
    Mean::compute<Device::CPU, T>(const StorageView& input,     \
                                  const dim_t outer_size,       \
                                  const dim_t axis_size,        \
                                  const dim_t inner_size,       \
                                  StorageView& output) const;

    DECLARE_IMPL(float)

  }
}
