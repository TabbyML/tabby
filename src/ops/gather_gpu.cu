#include "ctranslate2/ops/gather.h"

#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "cuda/helpers.h"
#include "cuda/utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    // Functor mapping output index to data index for Gather(axis=0, batch_dims=0).
    template <typename T>
    class batch_gather_index_map {
    private:
      const int32_t* _offsets;
      const T _stride;
    public:
      batch_gather_index_map(const int32_t* offsets, const T stride)
        : _offsets(offsets)
        , _stride(stride) {
      }
      __host__ __device__
      T operator()(const T i) const {
        const T row = i / _stride;
        const T col = i % _stride;
        return _offsets[row] * _stride + col;
      }
    };

    // Functor mapping output index to data index for Gather(axis=rank - 1, batch_dims=rank - 1).
    template <typename T>
    class depth_gather_index_map {
    private:
      const int32_t* _offsets;
      const T _depth;
      const T _gather_size;
    public:
      depth_gather_index_map(const int32_t* offsets,
                             const T depth,
                             const T gather_size)
        : _offsets(offsets)
        , _depth(depth)
        , _gather_size(gather_size) {
      }
      __host__ __device__
      T operator()(const T i) const {
        const T row = i / _gather_size;
        return row * _depth + _offsets[i];
      }
    };

    template <typename T, typename IndexMap>
    void run_gather(const IndexMap& index_map,
                    const T* src,
                    T* dst,
                    const dim_t dst_size) {
      auto gather_ids = thrust::make_transform_iterator(thrust::counting_iterator<cuda::index_t>(0),
                                                        index_map);
      THRUST_CALL(thrust::gather, gather_ids, gather_ids + dst_size, src, dst);
    }

    template <Device D, typename T>
    void Gather::compute(const StorageView& data,
                         const StorageView& input,
                         const dim_t axis,
                         const dim_t batch_dims,
                         StorageView& output) const {
      const dim_t dst_size = output.size();
      const int32_t* indices = input.data<int32_t>();
      const T* src = data.data<T>();
      T* dst = output.data<T>();

      if (axis == 0 && batch_dims == 0) {
        run_gather(batch_gather_index_map<cuda::index_t>(indices, data.stride(0)),
                   src, dst, dst_size);

      } else if (axis == data.rank() - 1 && batch_dims == data.rank() - 1) {
        const dim_t depth = data.dim(-1);
        const dim_t batch_size = data.size() / depth;
        const dim_t gather_size = input.size() / batch_size;  // Num. elements to gather per batch.
        run_gather(depth_gather_index_map<cuda::index_t>(indices, depth, gather_size),
                   src, dst, dst_size);

      } else {
        throw std::invalid_argument("unsupported gather configuration");
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Gather::compute<Device::CUDA, T>(const StorageView& data,           \
                                     const StorageView& input,          \
                                     const dim_t axis,                  \
                                     const dim_t batch_dims,            \
                                     StorageView& output) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
