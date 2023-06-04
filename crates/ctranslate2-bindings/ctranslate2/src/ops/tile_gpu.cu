#include "ctranslate2/ops/tile.h"

#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "cuda/helpers.h"
#include "cuda/utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    // Functor mapping an index in the tiled output to an input index.
    template <typename T>
    class tiled_index_map {
    private:
      const T _inner_size;
      const T _num_tiles;
    public:
      tiled_index_map(const T inner_size,
                      const T num_tiles)
        : _inner_size(inner_size)
        , _num_tiles(num_tiles)
      {
      }
      __device__
      T operator()(const T i) const {
        const T r = i / (_inner_size * _num_tiles);
        const T c = i % _inner_size;
        return r * _inner_size + c;
      }
    };

    template <Device D, typename T>
    void Tile::compute(const StorageView& input,
                       const dim_t,
                       const dim_t inner_size,
                       StorageView& output) const {
      auto gather_ids = thrust::make_transform_iterator(
        thrust::counting_iterator<cuda::index_t>(0),
        tiled_index_map<cuda::index_t>(inner_size, _num_tiles));
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
