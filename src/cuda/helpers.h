#pragma once

#include <algorithm>
#include <limits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "ctranslate2/types.h"

#include "utils.h"

#if !defined(__CUDACC__) || !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530
#  define CUDA_CAN_USE_HALF 1
#else
#  define CUDA_CAN_USE_HALF 0
#endif

#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
#  define CUDA_CAN_USE_BF16_MATH 1
#else
#  define CUDA_CAN_USE_BF16_MATH 0
#endif

namespace ctranslate2 {
  namespace cuda {

    // The index type used in CUDA kernels.
    // Currently set to a 32-bit type to maximize performance.
    using index_t = unsigned int;

    constexpr dim_t max_threads = 1024;
    constexpr dim_t max_blocks = std::numeric_limits<int32_t>::max();

    template <typename T>
    struct DeviceType {
      using type = T;
    };

    // Map float16_t and bfloat16_t to their corresponding device types.
    template<>
    struct DeviceType<float16_t> {
      using type = __half;
    };

    template<>
    struct DeviceType<bfloat16_t> {
      using type = __nv_bfloat16;
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
    inline void unary_transform(const T1* x, T2* y, index_t size, const UnaryFunction& op) {
      THRUST_CALL(thrust::transform, device_cast(x), device_cast(x) + size, device_cast(y), op);
    }

    template <typename T1, typename T2, typename T3, typename BinaryFunction>
    inline void binary_transform(const T1* a,
                                 const T2* b,
                                 T3* c,
                                 index_t size,
                                 const BinaryFunction& op) {
      THRUST_CALL(thrust::transform,
                  device_cast(a), device_cast(a) + size, device_cast(b), device_cast(c), op);
    }

    template <typename T1, typename T2, typename T3, typename BinaryFunction, typename IndexFunction>
    inline void binary_transform(const T1* a,
                                 const T2* b,
                                 T3* c,
                                 index_t size,
                                 const BinaryFunction& op,
                                 const IndexFunction& index_a) {
      auto index_it = thrust::make_transform_iterator(thrust::counting_iterator<index_t>(0), index_a);
      auto a_it = thrust::make_permutation_iterator(device_cast(a), index_it);
      THRUST_CALL(thrust::transform, a_it, a_it + size, device_cast(b), device_cast(c), op);
    }

    // perm_fun is a functor that takes the index in the permuted iterator and
    // return the index in the original iterator.
    template <typename T, typename PermFunction>
    inline void permute(const T* x, T* y, index_t size, const PermFunction& perm_fun) {
      auto ind_it = thrust::counting_iterator<index_t>(0);
      auto perm_ind_it = thrust::make_transform_iterator(ind_it, perm_fun);
      auto perm_it = thrust::make_permutation_iterator(device_cast(x), perm_ind_it);
      THRUST_CALL(thrust::copy, perm_it, perm_it + size, device_cast(y));
    }

    template <typename T>
    class repeat_vec {
    private:
      T _size;
    public:
      repeat_vec(T size)
        : _size(size) {
      }
      __device__
      T operator()(const T i) const {
        return i % _size;
      }
    };

    template <typename T>
    class repeat_vec_depth {
    private:
      T _size;
    public:
      repeat_vec_depth(T size)
        : _size(size) {
      }
      __device__
      T operator()(const T i) const {
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
      __device__ T operator()(const T& x) const {
        return _op(x, _y);
      }
    };

    // Some functional operators, similar to the ones from Thrust.

    template <typename T>
    struct plus {
      __device__ T operator()(const T& lhs, const T& rhs) const {
        return lhs + rhs;
      }
    };

    template <typename T>
    struct minus {
      __device__ T operator()(const T& lhs, const T& rhs) const {
        return lhs - rhs;
      }
    };

    template <typename T>
    struct multiplies {
      __device__ T operator()(const T& lhs, const T& rhs) const {
        return lhs * rhs;
      }
    };

    template <typename T>
    struct maximum {
      __device__ T operator()(const T& lhs, const T& rhs) const {
        return lhs < rhs ? rhs : lhs;
      }
    };

    template <typename T>
    struct minimum {
      __device__ T operator()(const T& lhs, const T& rhs) const {
        return lhs < rhs ? lhs : rhs;
      }
    };

#if !CUDA_CAN_USE_HALF
    template<>
    struct plus<__half> {
      __device__ __half operator()(const __half& lhs, const __half& rhs) const {
        return __half(float(lhs) + float(rhs));
      }
    };

    template<>
    struct minus<__half> {
      __device__ __half operator()(const __half& lhs, const __half& rhs) const {
        return __half(float(lhs) - float(rhs));
      }
    };

    template<>
    struct multiplies<__half> {
      __device__ __half operator()(const __half& lhs, const __half& rhs) const {
        return __half(float(lhs) * float(rhs));
      }
    };

    template<>
    struct maximum<__half> {
      __device__ __half operator()(const __half& lhs, const __half& rhs) const {
        return float(lhs) < float(rhs) ? rhs : lhs;
      }
    };

    template<>
    struct minimum<__half> {
      __device__ __half operator()(const __half& lhs, const __half& rhs) const {
        return float(lhs) < float(rhs) ? lhs : rhs;
      }
    };
#endif

    template <typename T>
    struct relu_func {
      __device__ T operator()(T x) const {
        return x > T(0.f) ? x : T(0.f);
      }
    };

#if !CUDA_CAN_USE_HALF
    template<>
    struct relu_func<__half> {
      __device__ __half operator()(__half x) const {
        return float(x) > float(0) ? x : __half(0);
      }
    };
#endif

    template <typename T>
    struct gelu_func {
      // Implicitly promote half to float in this function.
      __device__ float operator()(float x) const {
        return 0.5f * x * (1 + erff(0.7071067811865475f * x));
      }
    };

    template <typename T>
    struct gelu_tanh_func {
      // Implicitly promote half to float in this function.
      __device__ float operator()(float x) const {
        return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * powf(x, 3.f))));
      }
    };

    template <typename T>
    struct gelu_sigmoid_func {
      // Implicitly promote half to float in this function.
      __device__ float operator()(float x) const {
        return x / (1.f + expf(-1.702f * x));
      }
    };

    template <typename T>
    struct swish_func {
      // Implicitly promote half to float in this function.
      __device__ float operator()(float x) const {
        return x / (1.f + expf(-x));
      }
    };

    template <typename T>
    struct tanh_func {
      // Implicitly promote half to float in this function.
      __device__ float operator()(float x) const {
        return tanhf(x);
      }
    };

    template <typename T>
    struct sin_func {
      __device__ T operator()(T x) const {
        return sinf(x);
      }
    };

    template <typename T>
    struct cos_func {
      __device__ T operator()(T x) const {
        return cosf(x);
      }
    };

    template <typename T>
    struct exp_func {
      __device__ T operator()(T x) const {
        return expf(x);
      }
    };

    template <typename T>
    struct log_func {
      __device__ T operator()(T x) const {
        return logf(x);
      }
    };

#if CUDA_CAN_USE_HALF
    template<>
    struct sin_func<__half> {
      __device__ __half operator()(__half x) const {
        return hsin(x);
      }
    };

    template<>
    struct cos_func<__half> {
      __device__ __half operator()(__half x) const {
        return hcos(x);
      }
    };

    template<>
    struct exp_func<__half> {
      __device__ __half operator()(__half x) const {
        return hexp(x);
      }
    };

    template<>
    struct log_func<__half> {
      __device__ __half operator()(__half x) const {
        return hlog(x);
      }
    };
#endif

#if CUDA_CAN_USE_BF16_MATH
    template<>
    struct sin_func<__nv_bfloat16> {
      __device__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return hsin(x);
      }
    };

    template<>
    struct cos_func<__nv_bfloat16> {
      __device__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return hcos(x);
      }
    };

    template<>
    struct exp_func<__nv_bfloat16> {
      __device__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return hexp(x);
      }
    };

    template<>
    struct log_func<__nv_bfloat16> {
      __device__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return hlog(x);
      }
    };
#endif

    // The following kernels are adapted from:
    // https://github.com/pytorch/pytorch/blob/40eff454ce5638fbff638a7f4502e29ffb9a2f0d/aten/src/ATen/native/cuda/SoftMax.cu
    // They help define row-wise reduction where each block handles a single row.

#define C10_WARP_SIZE 32

    template <index_t ILP = 2>
    inline dim3 get_block_size(index_t dim_size) {
      index_t block_size = 1;
      index_t max_block_size = std::min(dim_size / ILP, static_cast<index_t>(max_threads));
      while (block_size < max_block_size)
        block_size *= 2;
      // Launch at least a single warp - the kernel assumes that.
      block_size = std::max(static_cast<index_t>(block_size), static_cast<index_t>(C10_WARP_SIZE));
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
        index_t lane = threadIdx.x % C10_WARP_SIZE;
        if (lane < blockDim.x / C10_WARP_SIZE) {
          #pragma unroll
          for (index_t i = 0; i < C10_WARP_SIZE; ++i) {
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
        for (index_t i = 0; i < blockDim.x / C10_WARP_SIZE; ++i) {
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
              index_t ILP = 2>
    __device__ __forceinline__ AccumT ilp_reduce(const T* data,
                                                 index_t size,
                                                 const Reduction& r,
                                                 AccumT defaultVal)
    {
      AccumT threadVal = defaultVal;
      index_t offset = threadIdx.x;
      index_t last = size % (ILP * blockDim.x);

      // Body (unroll by ILP times)
      for (; offset < size - last; offset += blockDim.x * ILP) {
        T tmp[ILP];

        #pragma unroll
        for (index_t j = 0; j < ILP; ++j)
          tmp[j] = data[offset + j * blockDim.x];

        #pragma unroll
        for (index_t j = 0; j < ILP; ++j)
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
              index_t ILP = 2>
    __device__ __forceinline__ void
    apply_epilogue(const scalar_t* input,
                   index_t depth,
                   const Epilogue& epilogue,
                   outscalar_t* output) {
      index_t offset = threadIdx.x;
      index_t last = depth % (ILP * blockDim.x);
      for (; offset < depth - last; offset += blockDim.x * ILP) {
        scalar_t tmp[ILP];

        #pragma unroll
        for (index_t j = 0; j < ILP; ++j)
          tmp[j] = input[offset + j * blockDim.x];

        #pragma unroll
        for (index_t j = 0; j < ILP; ++j)
          output[offset + j * blockDim.x] = epilogue(tmp[j]);
      }

      for (; offset < depth; offset += blockDim.x)
        output[offset] = epilogue(input[offset]);
    }

  }
}
