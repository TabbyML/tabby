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

}
