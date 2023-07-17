#include "ctranslate2/ops/softmax.h"

#include "cuda/helpers.h"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

    template <typename T>
    static void softmax_kernel(cudaStream_t stream,
                               const bool log_softmax,
                               const T* x,
                               const dim_t rows,
                               const dim_t cols,
                               const int32_t* lengths,
                               T* y);

    template <Device D, typename T>
    void SoftMax::compute(const StorageView& input,
                          const StorageView* lengths,
                          StorageView& output) const {
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      softmax_kernel(cuda::get_cuda_stream(),
                     _log,
                     input.data<T>(),
                     batch_size,
                     depth,
                     lengths ? lengths->data<int32_t>() : nullptr,
                     output.data<T>());
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    SoftMax::compute<Device::CUDA, T>(const StorageView& input,         \
                                      const StorageView* lengths,       \
                                      StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

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

    constexpr float max_float = std::numeric_limits<float>::max();

    template<typename T, typename AccumT, typename OutT>
    struct LogSoftMaxForwardEpilogue {
      __device__ __forceinline__ LogSoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
        : max_input(max_input),  logsum(std::log(sum)) {}

      __device__ __forceinline__ OutT operator()(T input) const {
        return static_cast<OutT>(static_cast<AccumT>(input) - max_input - logsum);
      }

      const AccumT max_input;
      const AccumT logsum;
    };

    template<typename T, typename AccumT, typename OutT>
    struct SoftMaxForwardEpilogue {
      __device__ __forceinline__ SoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
        : max_input(max_input)
        , sum(sum) {}

      __device__ __forceinline__ OutT operator()(T input) const {
        return static_cast<OutT>(std::exp(static_cast<AccumT>(input) - max_input) / sum);
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
        return sum + std::exp(static_cast<AccumT>(v) - max_k);
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

    template <typename scalar_t,
              typename accscalar_t,
              typename outscalar_t,
              typename index_t,
              typename length_t,
              template <typename, typename, typename> class Epilogue>
    __global__ void
    cunn_SoftMaxForward(outscalar_t *output,
                        const scalar_t *input,
                        const index_t classes,
                        const length_t *lengths)
    {
      extern __shared__ unsigned char smem[];
      auto sdata = reinterpret_cast<accscalar_t*>(smem);
      // forward pointers to batch[blockIdx.x]
      // each block handles a sample in the mini-batch
      const index_t row = blockIdx.x;
      input += row * classes;
      output += row * classes;

      index_t size = classes;
      if (lengths)
      {
        // Directly set 0 in output for out of range positions.
        size = lengths[row];
        for (index_t i = size + threadIdx.x; i < classes; i += blockDim.x)
          output[i] = 0.f;
      }

      // find the max
      accscalar_t threadMax = ctranslate2::cuda::ilp_reduce(
        input, size, MaxFloat<scalar_t, accscalar_t>(), -max_float);
      accscalar_t max_k = ctranslate2::cuda::block_reduce(
        sdata, threadMax, Max<accscalar_t>(), -max_float);

      // reduce all values
      accscalar_t threadExp = ctranslate2::cuda::ilp_reduce(
        input, size, SumExpFloat<scalar_t, accscalar_t>(max_k), static_cast<accscalar_t>(0));
      accscalar_t sumAll = ctranslate2::cuda::block_reduce(
        sdata, threadExp, Add<accscalar_t>(), static_cast<accscalar_t>(0));

      // apply epilogue
      ctranslate2::cuda::apply_epilogue(
        input, size, Epilogue<scalar_t, accscalar_t, outscalar_t>(max_k, sumAll), output);
    }

  }
}

namespace ctranslate2 {
  namespace ops {

    template <typename T, template <typename, typename, typename> class Epilogue>
    static void softmax_kernel_impl(cudaStream_t stream,
                                    const T* x,
                                    const dim_t rows,
                                    const dim_t cols,
                                    const int32_t* lengths,
                                    T* y) {
      const dim3 grid(rows);
      const dim3 block(cuda::get_block_size(cols));
      at::native::cunn_SoftMaxForward<T, float, T, cuda::index_t, int32_t, Epilogue>
        <<<grid, block, block.x * sizeof (float), stream>>>(y,
                                                            x,
                                                            cols,
                                                            lengths);
    }

    template <typename T>
    static void softmax_kernel(cudaStream_t stream,
                               const bool log_softmax,
                               const T* x,
                               const dim_t rows,
                               const dim_t cols,
                               const int32_t* lengths,
                               T* y) {
      if (log_softmax)
        softmax_kernel_impl<cuda::device_type<T>, at::native::LogSoftMaxForwardEpilogue>(
          stream, cuda::device_cast(x), rows, cols, lengths, cuda::device_cast(y));
      else
        softmax_kernel_impl<cuda::device_type<T>, at::native::SoftMaxForwardEpilogue>(
          stream, cuda::device_cast(x), rows, cols, lengths, cuda::device_cast(y));
    }

  }
}
