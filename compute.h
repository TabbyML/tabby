#pragma once

#include <algorithm>

#include <Eigen/Dense>

namespace opennmt {
  namespace compute {

    template <typename T>
    void fill(T* x, T a, size_t size) {
      std::fill_n(x, size, a);
    }

    template <typename T>
    void copy(const T* x, T* y, size_t size) {
      std::copy_n(x, size, y);
    }

    template <typename T>
    T sum(const T* array, size_t size) {
      return std::accumulate(array, array + size, static_cast<T>(0));
    }

    template <typename T>
    T mean(const T* array, size_t size) {
      return sum(array, size) / size;
    }

    template <typename T>
    size_t max_element(const T* array, size_t size) {
      return std::max_element(array, array + size) - array;
    }

    template <typename T>
    T max(const T* array, size_t size) {
      return array[max_element(array, size)];
    }

    template <typename T, typename I>
    void topk(const T* x, I* indices, size_t k, size_t size) {
      const auto comp = [&x](I i1, I i2) {
        return x[i1] > x[i2];
      };
      for (I i = 0; i < static_cast<I>(size); ++i)
        indices[i] = i;
      std::nth_element(indices, indices + k, indices + size, comp);
      std::sort(indices, indices + k, comp);
    }

    template <typename T>
    void add(T a, T* y, size_t size) {
      for (size_t i = 0; i < size; ++i)
        y[i] += a;
    }

    template <typename T>
    void add(const T* x, T* y, size_t size) {
      for (size_t i = 0; i < size; ++i)
        y[i] += x[i];
    }

    template <typename T>
    void add(const T* a, const T* b, T* c, size_t size) {
      for (size_t i = 0; i < size; ++i)
        c[i] = a[i] + b[i];
    }

    template <typename T>
    void sub(T a, T* y, size_t size) {
      T a_rev = -a;
      add(a_rev, y, size);
    }

    template <typename T>
    void mul(T a, T* y, size_t size) {
      for (size_t i = 0; i < size; ++i)
        y[i] *= a;
    }

    template <typename T>
    void mul(const T* x, T* y, size_t size) {
      for (size_t i = 0; i < size; ++i)
        y[i] *= x[i];
    }

    template <typename T>
    void mul(const T* a, const T* b, T* c, size_t size) {
      for (size_t i = 0; i < size; ++i)
        c[i] = a[i] * b[i];
    }

    template <typename T>
    void inv(const T* x, T* y, size_t size) {
      for (size_t i = 0; i < size; ++i)
        y[i] = static_cast<T>(1) / x[i];
    }

    template <typename In, typename Out>
    void quantize(const In* x, Out* y, size_t size, In scale, In shift) {
      for (size_t i = 0; i < size; ++i)
        y[i] = static_cast<Out>(x[i] * scale + shift);
    }

    template <typename In, typename Out>
    void unquantize(const In* x, Out* y, size_t size, Out scale, Out shift) {
      for (size_t i = 0; i < size; ++i)
        y[i] = (static_cast<Out>(x[i]) - shift) / scale;
    }

    template <typename T>
    void relu(T* x, size_t size) {
      for (size_t i = 0; i < size; ++i) {
        if (x[i] < static_cast<T>(0))
          x[i] = static_cast<T>(0);
      }
    }

    template <typename T>
    void relu(const T* x, T* y, size_t size) {
      for (size_t i = 0; i < size; ++i) {
        y[i] = x[i] > 0 ? x[i] : static_cast<T>(0);
      }
    }

    template <typename DataType, typename IndexType>
    void transpose_2d(const DataType* a, const IndexType* dims, DataType* b) {
      for (size_t i0 = 0; i0 < dims[0]; ++i0) {
        for (size_t i1 = 0; i1 < dims[1]; ++i1) {
          b[i1 * dims[0] + i0] = a[i0 * dims[1] + i1];
        }
      }
    }

    template <typename DataType, typename IndexType>
    void transpose_3d(const DataType* a,
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

    template <typename DataType, typename IndexType>
    void transpose_4d(const DataType* a,
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

    template <typename T>
    void pow(const T* x, T* y, T power, size_t size) {
      for (size_t i = 0; i < size; ++i) {
        y[i] = static_cast<T>(std::pow(static_cast<float>(x[i]), static_cast<float>(power)));
      }
    }

    template <typename T>
    void exp(const T* x, T* y, size_t size) {
      for (size_t i = 0; i < size; ++i) {
        y[i] = static_cast<T>(std::exp(x[i]));
      }
    }

    template <typename T>
    void cos(const T* x, T* y, size_t size) {
      for (size_t i = 0; i < size; ++i) {
        y[i] = static_cast<T>(std::cos(x[i]));
      }
    }

    template <typename T>
    void sin(const T* x, T* y, size_t size) {
      for (size_t i = 0; i < size; ++i) {
        y[i] = static_cast<T>(std::sin(x[i]));
      }
    }

    template <typename T>
    void tanh(const T* x, T* y, size_t size) {
      for (size_t i = 0; i < size; ++i) {
        y[i] = static_cast<T>(std::tanh(x[i]));
      }
    }

    template <typename In, typename Out>
    void gemm(const In* a, const In* b,
              bool transpose_a, bool transpose_b,
              size_t m, size_t n, size_t k,
              In alpha, Out beta,
              Out* c);

    template <typename T>
    void gemm(const T* a, const T* b,
              bool transpose_a, bool transpose_b,
              size_t m, size_t n, size_t k,
              T alpha, T beta,
              T* c) {
      Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> a_map(
        const_cast<T*>(a), transpose_a ? k : m, transpose_a ? m : k);
      Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> b_map(
        const_cast<T*>(b), transpose_b ? n : k, transpose_b ? k : n);
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> c_map(
        c, m, n);

      c_map *= beta;

      if (transpose_a && transpose_b) {
        c_map.noalias() += alpha * a_map.transpose() * b_map.transpose();
      } else if (transpose_a) {
        c_map.noalias() += alpha * a_map.transpose() * b_map;
      } else if (transpose_b) {
        c_map.noalias() += alpha * a_map * b_map.transpose();
      } else {
        c_map.noalias() += alpha * a_map * b_map;
      }
    }

    template <typename In, typename Out>
    void gemm_batch(const In* a, const In* b,
                    bool transpose_a, bool transpose_b,
                    size_t batch_size,
                    size_t m, size_t n, size_t k,
                    In alpha, Out beta,
                    Out* c) {
      for (size_t i = 0; i < batch_size; ++i) {
        const In* a_i = a + (i * m * k);
        const In* b_i = b + (i * k * n);
        Out* c_i = c + (i * m * n);

        gemm(a_i, b_i, transpose_a, transpose_b, m, n, k, alpha, beta, c_i);
      }
    }

  }
}
