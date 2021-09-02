#include "ctranslate2/ops/tile.h"

#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "cuda/utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    // Functor mapping an index in the tiled output to an input index.
    class tiled_index_map {
    private:
      const dim_t _inner_size;
      const dim_t _num_tiles;
    public:
      tiled_index_map(const dim_t inner_size,
                      const dim_t num_tiles)
        : _inner_size(inner_size)
        , _num_tiles(num_tiles)
      {
      }
      __host__ __device__
      dim_t operator()(const dim_t i) const {
        const dim_t r = i / (_inner_size * _num_tiles);
        const dim_t c = i % _inner_size;
        return r * _inner_size + c;
      }
    };

    template <Device D, typename T>
    void Tile::compute(const StorageView& input,
                       const dim_t,
                       const dim_t inner_size,
                       StorageView& output) const {
      auto gather_ids = thrust::make_transform_iterator(thrust::counting_iterator<dim_t>(0),
                                                        tiled_index_map(inner_size, _num_tiles));
      THRUST_CALL(thrust::gather,
                  gather_ids,
                  gather_ids + output.size(),
                  input.data<T>(),
                  output.data<T>());
    }

#define DECLARE_IMPL(T)                                          \
    template void                                                \
    Tile::compute<Device::CUDA, T>(const StorageView& input,     \
                                   const dim_t outer_size,       \
                                   const dim_t inner_size,       \
                                   StorageView& output) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
