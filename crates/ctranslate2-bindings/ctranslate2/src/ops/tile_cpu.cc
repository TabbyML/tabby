#include "ctranslate2/ops/tile.h"

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Tile::compute(const StorageView& input,
                       const dim_t outer_size,
                       const dim_t inner_size,
                       StorageView& output) const {
      const auto* src = input.data<T>();
      auto* dst = output.data<T>();

      for (dim_t i = 0; i < outer_size; ++i) {
        for (dim_t t = 0; t < _num_tiles; ++t) {
          primitives<D>::copy(src, dst, inner_size);
          dst += inner_size;
        }
        src += inner_size;
      }
    }

#define DECLARE_IMPL(T)                                         \
    template void                                               \
    Tile::compute<Device::CPU, T>(const StorageView& input,     \
                                  const dim_t outer_size,       \
                                  const dim_t inner_size,       \
                                  StorageView& output) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
