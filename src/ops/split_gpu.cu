#include "ctranslate2/ops/split.h"

#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "ctranslate2/cuda/utils.h"
#include "ctranslate2/cuda/concat_split_utils.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Split::compute(const StorageView& input,
                        std::vector<StorageView*>& outputs) const {
      size_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      size_t offset = 0;
      for (auto* output : outputs) {
        auto& x = *output;
        if (axis == 0) { // First outer dim.
          primitives<D>::copy(input.data<T>() + offset, x.data<T>(), x.size());
          offset += x.size();
        } else if (axis == input.rank() - 1) { // Last outer dim.
          auto gather_ids = thrust::make_transform_iterator(
            thrust::counting_iterator<size_t>(0),
            cuda::depth_select<int32_t>(offset, x.dim(-1), input.dim(-1)));
          thrust::gather(
            thrust::cuda::par.on(cuda::get_cuda_stream()),
            gather_ids, gather_ids + x.size(), input.data<T>(), x.data<T>());
          offset += x.dim(-1);
        } else { // Inner dim.
          size_t outer_dim = 1;
          for (size_t i = axis + 1; i < input.rank(); ++i)
            outer_dim *= input.dim(i);
          auto gather_ids = thrust::make_transform_iterator(
            thrust::counting_iterator<size_t>(0),
            cuda::inner_dim_select<int32_t>(offset, x.dim(axis), outer_dim, input.dim(axis)));
          thrust::gather(
            thrust::cuda::par.on(cuda::get_cuda_stream()),
            gather_ids, gather_ids + x.size(), input.data<T>(), x.data<T>());
          offset += x.dim(axis);
        }
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Split::compute<Device::CUDA, T>(const StorageView& input,           \
                                    std::vector<StorageView*>& outputs) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
