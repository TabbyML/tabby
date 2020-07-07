#include "ctranslate2/ops/concat.h"
#include "ctranslate2/ops/split.h"

#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scatter.h>

#include "cuda/utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <typename T>
    struct depth_select : public thrust::unary_function<T, T> {
      T _offset;
      T _depth;
      T _total_depth;
      depth_select(dim_t offset, dim_t depth, dim_t total_depth)
        : _offset(offset)
        , _depth(depth)
        , _total_depth(total_depth) {
      }
      __host__ __device__
      T operator()(const T& i) const {
        T row = i / _depth;
        T col = i % _depth;
        return row * _total_depth + col + _offset;
      }
    };

    template <typename T>
    struct inner_dim_select : public thrust::unary_function<T, T> {
      T _offset;
      T _inner_dim;
      T _outer_dim;
      T _total_inner_dim;
      inner_dim_select(dim_t offset,
                       dim_t inner_dim,
                       dim_t outer_dim,
                       dim_t total_inner_dim)
        : _offset(offset)
        , _inner_dim(inner_dim)
        , _outer_dim(outer_dim)
        , _total_inner_dim(total_inner_dim) {
      }
      __host__ __device__
      T operator()(const T& i) const {
        T i0 = i / (_inner_dim * _outer_dim);
        T i1 = (i / _outer_dim) % _inner_dim;
        T i2 = i % _outer_dim;
        return i0 * (_total_inner_dim * _outer_dim) + (i1 + _offset) * _outer_dim + i2;
      }
    };

    template <Device D, typename T>
    void Concat::compute(const std::vector<StorageView*>& inputs,
                         StorageView& output) const {
      const dim_t axis = _axis < 0 ? output.rank() + _axis : _axis;
      dim_t offset = 0;
      for (const auto& x : inputs) {
        if (axis == 0) {
          primitives<D>::copy(x->data<T>(), output.data<T>() + offset, x->size());
          offset += x->size();
        } else if (axis == output.rank() - 1) {
          auto map_ids = thrust::make_transform_iterator(
            thrust::counting_iterator<int32_t>(0),
            depth_select<int32_t>(offset, x->dim(-1), output.dim(-1)));
          THRUST_CALL(thrust::scatter,
                      x->data<T>(), x->data<T>() + x->size(), map_ids, output.data<T>());
          offset += x->dim(-1);
        } else {
          dim_t outer_dim = 1;
          for (dim_t i = axis + 1; i < output.rank(); ++i)
            outer_dim *= output.dim(i);
          auto map_ids = thrust::make_transform_iterator(
            thrust::counting_iterator<int32_t>(0),
            inner_dim_select<int32_t>(offset, x->dim(axis), outer_dim, output.dim(axis)));
          THRUST_CALL(thrust::scatter,
                      x->data<T>(), x->data<T>() + x->size(), map_ids, output.data<T>());
          offset += x->dim(axis);
        }
      }
    }

    template <Device D, typename T>
    void Split::compute(const StorageView& input,
                        std::vector<StorageView*>& outputs) const {
      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      dim_t offset = 0;
      for (auto* output : outputs) {
        auto& x = *output;
        if (axis == 0) { // First outer dim.
          primitives<D>::copy(input.data<T>() + offset, x.data<T>(), x.size());
          offset += x.size();
        } else if (axis == input.rank() - 1) { // Last outer dim.
          auto gather_ids = thrust::make_transform_iterator(
            thrust::counting_iterator<int32_t>(0),
            depth_select<int32_t>(offset, x.dim(-1), input.dim(-1)));
          THRUST_CALL(thrust::gather,
                      gather_ids, gather_ids + x.size(), input.data<T>(), x.data<T>());
          offset += x.dim(-1);
        } else { // Inner dim.
          dim_t outer_dim = 1;
          for (dim_t i = axis + 1; i < input.rank(); ++i)
            outer_dim *= input.dim(i);
          auto gather_ids = thrust::make_transform_iterator(
            thrust::counting_iterator<int32_t>(0),
            inner_dim_select<int32_t>(offset, x.dim(axis), outer_dim, input.dim(axis)));
          THRUST_CALL(thrust::gather,
                      gather_ids, gather_ids + x.size(), input.data<T>(), x.data<T>());
          offset += x.dim(axis);
        }
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Concat::compute<Device::CUDA, T>(const std::vector<StorageView*>& inputs, \
                                     StorageView& output) const;        \
    template void                                                       \
    Split::compute<Device::CUDA, T>(const StorageView& input,           \
                                    std::vector<StorageView*>& outputs) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
