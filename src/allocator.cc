#include "ctranslate2/allocator.h"

#include "device_dispatch.h"

namespace ctranslate2 {

  Allocator& get_allocator(Device device) {
    Allocator* allocator = nullptr;
    DEVICE_DISPATCH(device, allocator = &get_allocator<D>());
    if (!allocator)
      throw std::runtime_error("No allocator defined for device " + device_to_str(device));
    return *allocator;
  }

}
