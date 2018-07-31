#include "ctranslate2/ops/concat.h"

#include <thrust/scatter.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "ctranslate2/cuda/utils.h"
#include "ctranslate2/cuda/concat_split_utils.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Concat::compute(const std::vector<StorageView*>& inputs,
                         StorageView& output) const {
      size_t axis = _axis < 0 ? output.rank() + _axis : _axis;
      size_t offset = 0;
      for (const auto& x : inputs) {
        if (axis == 0) {
          primitives<D>::copy(x->data<T>(), output.data<T>() + offset, x->size());
          offset += x->size();
        } else if (axis == output.rank() - 1) {
          auto map_ids = thrust::make_transform_iterator(
            thrust::counting_iterator<size_t>(0),
            cuda::depth_select<int32_t>(offset, x->dim(-1), output.dim(-1)));
          thrust::scatter(
            thrust::cuda::par.on(cuda::get_cuda_stream()),
            x->data<T>(), x->data<T>() + x->size(), map_ids, output.data<T>());
          offset += x->dim(-1);
        } else {
          size_t outer_dim = 1;
          for (size_t i = axis + 1; i < output.rank(); ++i)
            outer_dim *= output.dim(i);
          auto map_ids = thrust::make_transform_iterator(
            thrust::counting_iterator<size_t>(0),
            cuda::inner_dim_select<int32_t>(offset, x->dim(axis), outer_dim, output.dim(axis)));
          thrust::scatter(
            thrust::cuda::par.on(cuda::get_cuda_stream()),
            x->data<T>(), x->data<T>() + x->size(), map_ids, output.data<T>());
          offset += x->dim(axis);
        }
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Concat::compute<Device::CUDA, T>(const std::vector<StorageView*>& inputs, \
                                     StorageView& output) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
