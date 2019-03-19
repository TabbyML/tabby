#include "ctranslate2/primitives/cpu_generic.h"

namespace ctranslate2 {

#ifndef WITH_MKL
  template<>
  void* primitives<Device::CPU>::alloc_data(size_t size) {
    return malloc(size);
  }

  template<>
  void primitives<Device::CPU>::free_data(void* data) {
    free(data);
  }

  template<>
  void primitives<Device::CPU>::clear_cache() {
  }
#endif

  template<>
  void primitives<Device::CPU>::rescale_output(const int32_t* x,
                                               const float* input_scales,
                                               const float* weight_scales,
                                               float* y,
                                               size_t batch_size,
                                               size_t depth) {
    #pragma omp parallel for
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < depth; ++j) {
        const auto index = j + i * depth;
        y[index] = static_cast<float>(x[index]) / (input_scales[i] * weight_scales[j]);
      }
    }
  }

}
