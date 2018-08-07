#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#include "primitives_decl.h"

namespace ctranslate2 {

  template <typename T1, typename T2, typename Function>
  void unary_transform(const T1* x, T2* y, size_t size, Function func) {
    std::transform(x, x + size, y, func);
  }

  template <typename T1, typename T2, typename Function>
  void binary_transform(const T1* a, const T1* b, T2* c, size_t size, Function func) {
    std::transform(a, a + size, b, c, func);
  }

  template<>
  template <typename T>
  T primitives<Device::CPU>::deref(const T* x, size_t index) {
    return x[index];
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::fill(T* x, T a, size_t size) {
    std::fill_n(x, size, a);
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::copy(const T* x, T* y, size_t size) {
    std::copy_n(x, size, y);
  }

  template<>
  template <typename T>
  T primitives<Device::CPU>::sum(const T* array, size_t size) {
    return std::accumulate(array, array + size, static_cast<T>(0));
  }

  template<>
  template <typename T>
  size_t primitives<Device::CPU>::max_element(const T* array, size_t size) {
    return std::distance(array, std::max_element(array, array + size));
  }

  template<>
  template <typename T>
  T primitives<Device::CPU>::max(const T* array, size_t size) {
    return *std::max_element(array, array + size);
  }

  template<>
  template <typename T, typename I>
  void primitives<Device::CPU>::topk(const T* x, T* values, I* indices, size_t k, size_t size) {
    static thread_local std::vector<I> full_indices;
    if (size > full_indices.size())
      full_indices.resize(size);

    I* full_indices_raw = full_indices.data();
    std::iota(full_indices_raw, full_indices_raw + size, 0);
    std::partial_sort(full_indices_raw, full_indices_raw + k, full_indices_raw + size,
                      [&x](const I& i1, const I& i2) {
                        return x[i1] > x[i2];
                      });

    for (size_t i = 0; i < k; ++i) {
      indices[i] = full_indices_raw[i];
      values[i] = x[indices[i]];
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::add(T a, const T* x, T* y, size_t size) {
    unary_transform(x, y, size, [&a](const T& v) { return v + a; });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::add(const T* a, const T* b, T* c, size_t size) {
    binary_transform(a, b, c, size, std::plus<T>());
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::add_batch_broadcast(const T* a, const T* b, T* c,
                                                    size_t a_size, size_t b_size) {
    size_t iter_size = b_size / a_size;
    for (size_t i = 0; i < iter_size; ++i) {
      size_t offset = i * a_size;
      add(a, b + offset, c + offset, a_size);
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::sub(const T* a, const T* b, T* c, size_t size) {
    binary_transform(a, b, c, size, std::minus<T>());
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::mul(T a, const T* x, T* y, size_t size) {
    unary_transform(x, y, size, [&a](const T& v) { return v * a; });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::mul(const T* a, const T* b, T* c, size_t size) {
    binary_transform(a, b, c, size, std::multiplies<T>());
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::mul_batch_broadcast(const T* a, const T* b, T* c,
                                                    size_t a_size, size_t b_size) {
    size_t iter_size = b_size / a_size;
    for (size_t i = 0; i < iter_size; ++i) {
      size_t offset = i * a_size;
      mul(a, b + offset, c + offset, a_size);
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::inv(const T* x, T* y, size_t size) {
    unary_transform(x, y, size, [](const T& v) { return static_cast<T>(1) / v; });
  }

  template<>
  template <typename In, typename Out>
  void primitives<Device::CPU>::quantize(const In* x, Out* y, size_t size, In scale, In shift) {
    unary_transform(x, y, size, [&scale, &shift](const In& v) {
      return static_cast<Out>(v * scale + shift);
    });
  }

  template<>
  template <typename In, typename Out>
  void primitives<Device::CPU>::unquantize(const In* x, Out* y, size_t size, Out scale, Out shift) {
    unary_transform(x, y, size, [&scale, &shift](const In& v) {
      return (static_cast<Out>(v) - shift) / scale;
    });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::relu(const T* x, T* y, size_t size) {
    unary_transform(x, y, size, [](const T& v) {
      return v > 0 ? v : static_cast<T>(0);
    });
  }

  template<>
  template <typename DataType, typename IndexType>
  void primitives<Device::CPU>::transpose_2d(const DataType* a, const IndexType* dims, DataType* b) {
    for (size_t i0 = 0; i0 < dims[0]; ++i0) {
      for (size_t i1 = 0; i1 < dims[1]; ++i1) {
        b[i1 * dims[0] + i0] = a[i0 * dims[1] + i1];
      }
    }
  }

  template<>
  template <typename DataType, typename IndexType>
  void primitives<Device::CPU>::transpose_3d(const DataType* a,
                                             const IndexType* dims,
                                             const IndexType* perm,
                                             DataType* b) {
    size_t perm_ind[3];
    for (size_t i = 0; i < 3; ++i)
      perm_ind[perm[i]] = i;
    size_t a_stride[3] = {dims[1] * dims[2], dims[2], 1};
    size_t b_stride[3] = {dims[perm[1]] * dims[perm[2]], dims[perm[2]], 1};
    size_t perm_b_stride[3] = {b_stride[perm_ind[0]], b_stride[perm_ind[1]],
                               b_stride[perm_ind[2]]};

    for (size_t i0 = 0; i0 < dims[0]; ++i0) {
      for (size_t i1 = 0; i1 < dims[1]; ++i1) {
        for (size_t i2 = 0; i2 < dims[2]; ++i2) {
          const size_t b_i = (i0 * perm_b_stride[0] + i1 * perm_b_stride[1] +
                              i2 * perm_b_stride[2]);
          const size_t a_i = (i0 * a_stride[0] + i1 * a_stride[1] +
                              i2 * a_stride[2]);
          b[b_i] = a[a_i];
        }
      }
    }
  }

  template<>
  template <typename DataType, typename IndexType>
  void primitives<Device::CPU>::transpose_4d(const DataType* a,
                                             const IndexType* dims,
                                             const IndexType* perm,
                                             DataType* b) {
    size_t perm_ind[4];
    for (size_t i = 0; i < 4; ++i)
      perm_ind[perm[i]] = i;
    size_t a_stride[4] = {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
    size_t b_stride[4] = {dims[perm[1]] * dims[perm[2]] * dims[perm[3]],
                          dims[perm[2]] * dims[perm[3]], dims[perm[3]], 1};
    size_t perm_b_stride[4] = {b_stride[perm_ind[0]], b_stride[perm_ind[1]],
                               b_stride[perm_ind[2]], b_stride[perm_ind[3]]};

    for (size_t i0 = 0; i0 < dims[0]; ++i0) {
      for (size_t i1 = 0; i1 < dims[1]; ++i1) {
        for (size_t i2 = 0; i2 < dims[2]; ++i2) {
          for (size_t i3 = 0; i3 < dims[3]; ++i3) {
            const size_t b_i = (i0 * perm_b_stride[0] + i1 * perm_b_stride[1] +
                                i2 * perm_b_stride[2] + i3 * perm_b_stride[3]);
            const size_t a_i = (i0 * a_stride[0] + i1 * a_stride[1] +
                                i2 * a_stride[2] + i3 * a_stride[3]);
            b[b_i] = a[a_i];
          }
        }
      }
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::pow(const T* x, T* y, T power, size_t size) {
    unary_transform(x, y, size, [&power](const T& v) {
      return static_cast<T>(std::pow(static_cast<float>(v), static_cast<float>(power)));
    });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::exp(const T* x, T* y, size_t size) {
    unary_transform(x, y, size, [](const T& v) { return static_cast<T>(std::exp(v)); });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::cos(const T* x, T* y, size_t size) {
    unary_transform(x, y, size, [](const T& v) { return static_cast<T>(std::cos(v)); });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::sin(const T* x, T* y, size_t size) {
    unary_transform(x, y, size, [](const T& v) { return static_cast<T>(std::sin(v)); });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::tanh(const T* x, T* y, size_t size) {
    unary_transform(x, y, size, [](const T& v) { return static_cast<T>(std::tanh(v)); });
  }

}
