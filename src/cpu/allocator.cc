#include "ctranslate2/allocator.h"

#ifdef _WIN32
#  include <malloc.h>
#else
#  include <cstdlib>
#endif

#ifdef CT2_WITH_MKL
#  include <mkl.h>
#endif

namespace ctranslate2 {
  namespace cpu {

    class AlignedAllocator : public Allocator {
    public:
      AlignedAllocator(size_t alignment)
        : _alignment(alignment)
      {
      }

      void* allocate(size_t size, int) override {
        void* ptr = nullptr;
#ifdef _WIN32
        ptr = _aligned_malloc(size, _alignment);
#else
        if (posix_memalign(&ptr, _alignment, size) != 0)
          ptr = nullptr;
#endif
        if (!ptr)
          throw std::runtime_error("aligned_alloc: failed to allocate memory");
        return ptr;
      }

      void free(void* ptr, int) override {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
      }

    private:
      size_t _alignment;
    };

#ifdef CT2_WITH_MKL
    class MklAllocator : public Allocator {
    public:
      MklAllocator(size_t alignment)
        : _alignment(alignment)
      {
      }

      void* allocate(size_t size, int) override {
        void* ptr = mkl_malloc(size, _alignment);
        if (!ptr)
          throw std::runtime_error("mkl_malloc: failed to allocate memory");
        return ptr;
      }

      void free(void* ptr, int) override {
        mkl_free(ptr);
      }

      void clear_cache() override {
        mkl_free_buffers();
      }

    private:
      size_t _alignment;
    };
#endif

  }

  template<>
  Allocator& get_allocator<Device::CPU>() {
    constexpr size_t alignment = 64;
#ifdef CT2_WITH_MKL
    static cpu::MklAllocator allocator(alignment);
#else
    static cpu::AlignedAllocator allocator(alignment);
#endif
    return allocator;
  }

}
