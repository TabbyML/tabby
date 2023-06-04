#pragma once

#include "devices.h"

namespace ctranslate2 {

  class Allocator {
  public:
    virtual ~Allocator() = default;

    virtual void* allocate(size_t size, int device_index) = 0;
    virtual void free(void* ptr, int device_index) = 0;
    virtual void clear_cache() {};

    void* allocate(size_t size) {
      return allocate(size, -1);
    }

    void free(void* ptr) {
      free(ptr, -1);
    }
  };

  template <Device D>
  Allocator& get_allocator();
  Allocator& get_allocator(Device device);

}
