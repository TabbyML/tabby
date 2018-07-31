#pragma once

#include <thrust/functional.h>

namespace ctranslate2 {
  namespace cuda {

    template <typename T>
    struct depth_select : public thrust::unary_function<T, T> {
      T _offset;
      T _depth;
      T _total_depth;
      depth_select(size_t offset, size_t depth, size_t total_depth)
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
      inner_dim_select(size_t offset,
                       size_t inner_dim,
                       size_t outer_dim,
                       size_t total_inner_dim)
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

  }
}
