#include "ctranslate2/ops/gather.h"

#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "cuda/utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    struct map_id : public thrust::unary_function<int32_t, int32_t> {
      const int32_t* _offsets;
      int32_t _stride;
      map_id(const int32_t* offsets, int32_t stride)
        : _offsets(offsets)
        , _stride(stride) {
      }
      __host__ __device__
      int32_t operator()(const int32_t& i) const {
        int32_t row = i / _stride;
        int32_t col = i % _stride;
        return _offsets[row] * _stride + col;
      }
    };

    template <Device D, typename T>
    void Gather::compute(const StorageView& data,
                         const StorageView& input,
                         StorageView& output) const {
      auto gather_ids = thrust::make_transform_iterator(
        thrust::counting_iterator<int32_t>(0),
        map_id(input.data<int32_t>(), data.stride(0)));
      THRUST_CALL(thrust::gather,
                  gather_ids, gather_ids + output.size(), data.data<T>(), output.data<T>());
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Gather::compute<Device::CUDA, T>(const StorageView& data,           \
                                     const StorageView& input,          \
                                     StorageView& output) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
