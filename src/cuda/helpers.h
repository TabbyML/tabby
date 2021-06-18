#pragma once

#include <algorithm>

#include <cuda_fp16.h>

#include "ctranslate2/types.h"

#include "utils.h"
#include "type_dispatch.h"

#if !defined(__CUDACC__) || !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530
#  define CUDA_CAN_USE_HALF 1
#else
#  define CUDA_CAN_USE_HALF 0
#endif

namespace ctranslate2 {
  namespace cuda {

    constexpr dim_t max_threads = 1024;
    constexpr dim_t max_blocks = 65535;

    template <typename T>
    struct DeviceType {
      using type = T;
    };

    // float16_t (a.k.a. half_float::half) can't be used in device code so there is a bit
    // of template work to cast values and pointers to __half.
    template<>
    struct DeviceType<float16_t> {
      using type = __half;
    };

    template <typename T>
    using device_type = typename DeviceType<T>::type;

    template <typename T>
    inline const device_type<T>* device_cast(const T* x) {
      return reinterpret_cast<const device_type<T>*>(x);
    }

    template <typename T>
    inline device_type<T>* device_cast(T* x) {
      return reinterpret_cast<device_type<T>*>(x);
    }

    template <typename T1, typename T2, typename UnaryFunction>
    inline void unary_transform(const T1* x, T2* y, dim_t size, const UnaryFunction& op) {
      THRUST_CALL(thrust::transform, device_cast(x), device_cast(x) + size, device_cast(y), op);
    }

    template <typename T1, typename T2, typename T3, typename BinaryFunction>
    inline void binary_transform(const T1* a,
                                 const T2* b,
                                 T3* c,
                                 dim_t size,
                                 const BinaryFunction& op) {
      THRUST_CALL(thrust::transform,
                  device_cast(a), device_cast(a) + size, device_cast(b), device_cast(c), op);
    }

    template <typename T1, typename T2, typename T3, typename BinaryFunction, typename IndexFunction>
    inline void binary_transform(const T1* a,
                                 const T2* b,
                                 T3* c,
                                 dim_t size,
                                 const BinaryFunction& op,
                                 const IndexFunction& index_a) {
      auto index_it = thrust::make_transform_iterator(thrust::counting_iterator<dim_t>(0), index_a);
      auto a_it = thrust::make_permutation_iterator(device_cast(a), index_it);
      THRUST_CALL(thrust::transform, a_it, a_it + size, device_cast(b), device_cast(c), op);
    }

    // perm_fun is a functor that takes the index in the permuted iterator and
    // return the index in the original iterator.
    template <typename T, typename PermFunction>
    inline void permute(const T* x, T* y, dim_t size, const PermFunction& perm_fun) {
      auto ind_it = thrust::counting_iterator<dim_t>(0);
      auto perm_ind_it = thrust::make_transform_iterator(ind_it, perm_fun);
      auto perm_it = thrust::make_permutation_iterator(device_cast(x), perm_ind_it);
      THRUST_CALL(thrust::copy, perm_it, perm_it + size, device_cast(y));
    }

    template <typename T>
    struct repeat_vec {
      T _size;
      repeat_vec(T size)
        : _size(size) {
      }
      __host__ __device__
      T operator()(const T i) {
        return i % _size;
      }
    };

    template <typename T>
    struct repeat_vec_depth {
      T _size;
      repeat_vec_depth(T size)
        : _size(size) {
      }
      __host__ __device__
      T operator()(const T i) {
        return i / _size;
      }
    };

    // Bind the right argument of a binary operator.
    template <template <typename> class BinaryFunctor, typename T>
    class bind_right {
    private:
      const T _y;
      BinaryFunctor<T> _op;
    public:
      bind_right(const T& y)
        : _y(y) {
      }
      __host__ __device__ T operator()(const T& x) const {
        return _op(x, _y);
      }
    };

    // Some functional operators, similar to the ones from Thrust.

    template <typename T>
    struct plus {
      __host__ __device__ T operator()(const T& lhs, const T& rhs) const {
        return lhs + rhs;
      }
    };

    template <typename T>
    struct minus {
      __host__ __device__ T operator()(const T& lhs, const T& rhs) const {
        return lhs - rhs;
      }
    };

    template <typename T>
    struct multiplies {
      __host__ __device__ T operator()(const T& lhs, const T& rhs) const {
        return lhs * rhs;
      }
    };

    template <typename T>
    struct maximum {
      __host__ __device__ T operator()(const T& lhs, const T& rhs) const {
        return lhs < rhs ? rhs : lhs;
      }
    };

    template <typename T>
    struct minimum {
      __host__ __device__ T operator()(const T& lhs, const T& rhs) const {
        return lhs < rhs ? lhs : rhs;
      }
    };

#if !CUDA_CAN_USE_HALF
    template<>
    struct plus<__half> {
      __host__ __device__ __half operator()(const __half& lhs, const __half& rhs) const {
        return __half(float(lhs) + float(rhs));
      }
    };

    template<>
    struct minus<__half> {
      __host__ __device__ __half operator()(const __half& lhs, const __half& rhs) const {
        return __half(float(lhs) - float(rhs));
      }
    };

    template<>
    struct multiplies<__half> {
      __host__ __device__ __half operator()(const __half& lhs, const __half& rhs) const {
        return __half(float(lhs) * float(rhs));
      }
    };

    template<>
    struct maximum<__half> {
      __host__ __device__ __half operator()(const __half& lhs, const __half& rhs) const {
        return float(lhs) < float(rhs) ? rhs : lhs;
      }
    };

    template<>
    struct minimum<__half> {
      __host__ __device__ __half operator()(const __half& lhs, const __half& rhs) const {
        return float(lhs) < float(rhs) ? lhs : rhs;
      }
    };
#endif

    template <typename T>
    struct relu_func : public bind_right<maximum, T> {
      relu_func() : bind_right<maximum, T>(T(0)) {}
    };

    class gelu_func {
    private:
      float _scale;
    public:
      gelu_func() : _scale(std::sqrt(2.f / std::acos(-1.f))) {}

      __host__ __device__
      float operator()(float x) {
        return 0.5f * x * (1.f + tanhf(_scale * (x + 0.044715f * powf(x, 3.f))));
      }
    };

    // The following kernels are adapted from:
    // https://github.com/pytorch/pytorch/blob/40eff454ce5638fbff638a7f4502e29ffb9a2f0d/aten/src/ATen/native/cuda/SoftMax.cu
    // They help define row-wise reduction where each block handles a single row.

#define C10_WARP_SIZE 32

    template <int ILP = 2>
    inline dim3 get_block_size(dim_t dim_size) {
      dim_t block_size = 1;
      dim_t max_block_size = std::min(dim_size / ILP, max_threads);
      while (block_size < max_block_size)
        block_size *= 2;
      // Launch at least a single warp - the kernel assumes that.
      block_size = std::max(block_size, static_cast<dim_t>(C10_WARP_SIZE));
      return dim3(block_size);
    }

    template <typename Reduction, typename AccumT>
    __device__ __forceinline__ AccumT block_reduce(AccumT* smem,
                                                   AccumT val,
                                                   const Reduction& r,
                                                   AccumT defaultVal)
    {
      // To avoid RaW races from chaining blockReduce calls together, we need a sync here
      __syncthreads();

      smem[threadIdx.x] = val;

      __syncthreads();

      AccumT warpVal = defaultVal;

      // First warp will perform per-warp reductions for the remaining warps
      uint32_t mask = (((uint64_t)1) << (blockDim.x / C10_WARP_SIZE)) - 1;
      if (threadIdx.x < C10_WARP_SIZE) {
        int lane = threadIdx.x % C10_WARP_SIZE;
        if (lane < blockDim.x / C10_WARP_SIZE) {
          #pragma unroll
          for (int i = 0; i < C10_WARP_SIZE; ++i) {
            warpVal = r(warpVal, smem[lane * C10_WARP_SIZE + i]);
          }
          __syncwarp(mask);
          smem[lane] = warpVal;
        }
      }

      __syncthreads();

      // First thread will perform a reduction of the above per-warp reductions
      AccumT blockVal = defaultVal;

      if (threadIdx.x == 0) {
        for (int i = 0; i < blockDim.x / C10_WARP_SIZE; ++i) {
          blockVal = r(blockVal, smem[i]);
        }
        smem[0] = blockVal;
      }

      // Sync and broadcast
      __syncthreads();
      return smem[0];
    }

    template <typename Reduction,
              typename T,
              typename AccumT = T,
              int ILP = 2>
    __device__ __forceinline__ AccumT ilp_reduce(const T* data,
                                                 int size,
                                                 const Reduction& r,
                                                 AccumT defaultVal)
    {
      AccumT threadVal = defaultVal;
      int offset = threadIdx.x;

      int last = size % (ILP * blockDim.x);

      // Body (unroll by ILP times)
      for (; offset < size - last; offset += blockDim.x * ILP) {
        T tmp[ILP];

        #pragma unroll
        for (int j = 0; j < ILP; ++j)
          tmp[j] = data[offset + j * blockDim.x];

        #pragma unroll
        for (int j = 0; j < ILP; ++j)
          threadVal = r(threadVal, tmp[j]);
      }

      // Epilogue
      for (; offset < size; offset += blockDim.x)
        threadVal = r(threadVal, data[offset]);

      return threadVal;
    }

    template <typename Epilogue,
              typename scalar_t,
              typename outscalar_t,
              int ILP = 2>
    __device__ __forceinline__ void
    apply_epilogue(const scalar_t* input,
                   int depth,
                   const Epilogue& epilogue,
                   outscalar_t* output) {
      int offset = threadIdx.x;
      int last = depth % (ILP * blockDim.x);
      for (; offset < depth - last; offset += blockDim.x * ILP) {
        scalar_t tmp[ILP];

        #pragma unroll
        for (int j = 0; j < ILP; ++j)
          tmp[j] = input[offset + j * blockDim.x];

        #pragma unroll
        for (int j = 0; j < ILP; ++j)
          output[offset + j * blockDim.x] = epilogue(tmp[j]);
      }

      for (; offset < depth; offset += blockDim.x)
        output[offset] = epilogue(input[offset]);
    }

  }
}
