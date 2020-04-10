#include "ctranslate2/ops/softmax.h"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/replace.h>

#include "../cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

    static void softmax_kernel(cudaStream_t stream,
                               const bool log_softmax,
                               const float* x,
                               const int64_t rows,
                               const int64_t cols,
                               float* y);

    // Operator returning true for each out of range positions.
    class mask_func {
    private:
      const int32_t* _lengths;
      const int32_t _batch_size;       // Batch size.
      const int32_t _flat_batch_size;  // Batch size * inner dimensions.
      const int32_t _depth;            // Last dimension.

    public:
      mask_func(const int32_t* lengths,
                int32_t batch_size,
                int32_t flat_batch_size,
                int32_t depth)
        : _lengths(lengths)
        , _batch_size(batch_size)
        , _flat_batch_size(flat_batch_size)
        , _depth(depth) {
      }

      __device__
      bool operator()(int32_t index) const {
        auto position = index % _depth;
        auto flat_batch = index / _depth;
        auto true_batch = flat_batch * _batch_size / _flat_batch_size;
        return position >= _lengths[true_batch];
      }
    };

    template <Device D, typename T>
    void SoftMax::compute(const StorageView& input,
                          const StorageView* lengths,
                          StorageView& output) const {
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;

      StorageView masked_input(input.device());
      const auto* data = input.data<float>();
      if (lengths) {
        masked_input.resize_as(input);
        auto* masked_data = masked_input.data<float>();

        // Copy input but replace out of range positions with -inf.
        THRUST_CALL(thrust::replace_copy_if,
                    data,
                    data + input.size(),
                    thrust::counting_iterator<int32_t>(0),
                    masked_data,
                    mask_func(lengths->data<int32_t>(), lengths->dim(0), batch_size, depth),
                    std::numeric_limits<float>::lowest());

        data = masked_data;
      }

      softmax_kernel(cuda::get_cuda_stream(),
                     _log,
                     data,
                     batch_size,
                     depth,
                     output.data<T>());
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    SoftMax::compute<Device::CUDA, T>(const StorageView& input,         \
                                      const StorageView* lengths,       \
                                      StorageView& output) const;

    DECLARE_IMPL(float)

  }
}

// The following CUDA kernels are adapted from:
// https://github.com/pytorch/pytorch/blob/40eff454ce5638fbff638a7f4502e29ffb9a2f0d/aten/src/ATen/native/cuda/SoftMax.cu
// which has the following license notice:

/*
  From PyTorch:

  Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
  Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
  Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
  Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
  Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
  Copyright (c) 2011-2013 NYU                      (Clement Farabet)
  Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
  Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
  Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

  From Caffe2:

  Copyright (c) 2016-present, Facebook Inc. All rights reserved.

  All contributions by Facebook:
  Copyright (c) 2016 Facebook Inc.

  All contributions by Google:
  Copyright (c) 2015 Google Inc.
  All rights reserved.

  All contributions by Yangqing Jia:
  Copyright (c) 2015 Yangqing Jia
  All rights reserved.

  All contributions from Caffe:
  Copyright(c) 2013, 2014, 2015, the respective contributors
  All rights reserved.

  All other contributions:
  Copyright(c) 2015, 2016 the respective contributors
  All rights reserved.

  Caffe2 uses a copyright model similar to Caffe: each contributor holds
  copyright over their contributions to Caffe2. The project versioning records
  all such contribution and copyright details. If a contributor wants to further
  mark their specific copyright on a particular contribution, they should
  indicate their copyright solely in the commit message of the change when it is
  committed.

  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

  3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
     and IDIAP Research Institute nor the names of its contributors may be
     used to endorse or promote products derived from this software without
     specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.
*/

namespace at {
  namespace native {

#ifdef __HIP_PLATFORM_HCC__
#  define C10_WARP_SIZE 64
#else
#  define C10_WARP_SIZE 32
#endif

    constexpr int max_threads = 1024;
    constexpr float max_float = std::numeric_limits<float>::max();

    static dim3 SoftMax_getBlockSize(int ILP, uint64_t dim_size) {
      uint64_t block_size = 1;
      uint64_t max_block_size = std::min(dim_size / ILP, static_cast<uint64_t>(max_threads));
      while (block_size < max_block_size) block_size *= 2;
      // Launch at least a single warp - the kernel assumes that.
      block_size = std::max(block_size, static_cast<uint64_t>(C10_WARP_SIZE));
      return dim3(block_size);
    }

    template<typename T, typename AccumT, typename OutT>
    struct LogSoftMaxForwardEpilogue {
      __device__ __forceinline__ LogSoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
        : logsum(max_input + std::log(sum)) {}

      __device__ __forceinline__ OutT operator()(T input) const {
        return static_cast<OutT>(input - logsum);
      }

      const AccumT logsum;
    };

    template<typename T, typename AccumT, typename OutT>
    struct SoftMaxForwardEpilogue {
      __device__ __forceinline__ SoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
        : max_input(max_input)
        , sum(sum) {}

      __device__ __forceinline__ OutT operator()(T input) const {
        return static_cast<OutT>(std::exp(input - max_input) / sum);
      }

      const AccumT max_input;
      const AccumT sum;
    };

    template<typename T, typename AccumT>
    struct SumExpFloat
    {
      __device__ __forceinline__ SumExpFloat(AccumT v)
        : max_k(v) {}

      __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
        return sum + std::exp(v - max_k);
      }

      const AccumT max_k;
    };

    template<typename T>
    struct Max {
      __device__ __forceinline__ T operator()(T a, T b) const {
        return a < b ? b : a;
      }
    };

    template <typename T, typename AccumT>
    struct MaxFloat
    {
      __device__ __forceinline__ AccumT operator()(AccumT max, T v) const {
        return ::max(max, (AccumT)v);
      }
    };

    template<typename T>
    struct Add {
      __device__ __forceinline__ T operator()(T a, T b) const {
        return a + b;
      }
    };

    template <template<typename> class Reduction, typename AccumT>
    __device__ __forceinline__ AccumT
    blockReduce(AccumT* smem, AccumT val,
                const Reduction<AccumT>& r,
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
#ifndef __HIP_PLATFORM_HCC__
          __syncwarp(mask);
#endif
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

    template <template<typename, typename> class Reduction, int ILP, typename T, typename AccumT>
    __device__ __forceinline__ AccumT
    ilpReduce(const T* data,
              int size,
              const Reduction<T, AccumT>& r,
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

    template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t, template <typename, typename, typename> class Epilogue>
    __global__ void
    cunn_SoftMaxForward(outscalar_t *output, const scalar_t *input, int classes)
    {
      extern __shared__ unsigned char smem[];
      auto sdata = reinterpret_cast<accscalar_t*>(smem);
      // forward pointers to batch[blockIdx.x]
      // each block handles a sample in the mini-batch
      input += blockIdx.x * classes;
      output += blockIdx.x * classes;

      // find the max
      accscalar_t threadMax = ilpReduce<MaxFloat, ILP, scalar_t, accscalar_t>(
        input, classes, MaxFloat<scalar_t, accscalar_t>(), -max_float);
      accscalar_t max_k = blockReduce<Max, accscalar_t>(
        sdata, threadMax, Max<accscalar_t>(), -max_float);

      // reduce all values
      accscalar_t threadExp = ilpReduce<SumExpFloat, ILP, scalar_t, accscalar_t>(
        input, classes, SumExpFloat<scalar_t, accscalar_t>(max_k), static_cast<accscalar_t>(0));
      accscalar_t sumAll = blockReduce<Add, accscalar_t>(
        sdata, threadExp, Add<accscalar_t>(), static_cast<accscalar_t>(0));

      Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(max_k, sumAll);
      int offset = threadIdx.x;
      int last = classes % (ILP * blockDim.x);
      for (; offset < classes - last; offset += blockDim.x * ILP) {
        scalar_t tmp[ILP];

        #pragma unroll
        for (int j = 0; j < ILP; ++j)
          tmp[j] = input[offset + j * blockDim.x];

        #pragma unroll
        for (int j = 0; j < ILP; ++j)
          output[offset + j * blockDim.x] = epilogue(tmp[j]);
      }

      for (; offset < classes; offset += blockDim.x)
        output[offset] = epilogue(input[offset]);
    }

  }
}

namespace ctranslate2 {
  namespace ops {

    template <template <typename, typename, typename> class Epilogue>
    static void softmax_kernel_impl(cudaStream_t stream,
                                    const float* x,
                                    const int64_t rows,
                                    const int64_t cols,
                                    float* y) {
      const int ILP = 2;
      const dim3 grid(rows);
      const dim3 block = at::native::SoftMax_getBlockSize(ILP, cols);
      at::native::cunn_SoftMaxForward<ILP, float, float, float, Epilogue>
        <<<grid, block, block.x * sizeof (float), stream>>>(y, x, cols);
    }

    static void softmax_kernel(cudaStream_t stream,
                               const bool log_softmax,
                               const float* x,
                               const int64_t rows,
                               const int64_t cols,
                               float* y) {
      if (log_softmax)
        softmax_kernel_impl<at::native::LogSoftMaxForwardEpilogue>(stream, x, rows, cols, y);
      else
        softmax_kernel_impl<at::native::SoftMaxForwardEpilogue>(stream, x, rows, cols, y);
    }

  }
}
