#pragma once

#include <cstddef>
#include <cstdint>

#ifdef INTEL_MKL
#  include <mkl.h>
#endif

namespace opennmt
{

  inline void *align( std::size_t alignment, std::size_t size,
                      void *&ptr, std::size_t &space ) {
    // Copyright 2014 David Krauss
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=57350
    std::uintptr_t pn = reinterpret_cast< std::uintptr_t >( ptr );
    std::uintptr_t aligned = ( pn + alignment - 1 ) & - alignment;
    std::size_t padding = aligned - pn;
    if ( space < size + padding ) return nullptr;
    space -= padding;
    return ptr = reinterpret_cast< void * >( aligned );
  }

  inline void init(size_t num_threads) {
#ifdef WITH_MKL
    vmlSetMode(VML_EP);
    if (num_threads > 0)
      mkl_set_num_threads(num_threads);
#endif
  }

}
