#include "ctranslate2/ops/layer_norm.h"

#include "../cuda/utils.h"

namespace at {
  namespace native {

    // Forward declaration of the CUDA kernels.
    __global__ void RowwiseMomentsCUDAKernel(int64_t N,
                                             float eps,
                                             const float* X,
                                             float* mean,
                                             float* rstd);
    __global__ void LayerNormForwardCUDAKernel(int64_t N,
                                               const float* X,
                                               const float* mean,
                                               const float* rstd,
                                               const float* gamma,
                                               const float* beta,
                                               float* Y);

  }
}

namespace ctranslate2 {
  namespace ops {

#define CUDA_NUM_THREADS 256
#define CUDA_BLOCK_REDUCE_NUM_THREADS 512
#define EPSILON 1e-4

    template <Device D, typename T>
    void LayerNorm::compute(const StorageView& beta,
                            const StorageView& gamma,
                            const StorageView& input,
                            StorageView& output) const {
      size_t depth = input.dim(-1);
      size_t batch_size = input.size() / depth;

      StorageView moments({2 * batch_size}, input.dtype(), input.device());
      T* mean_data = moments.data<T>();
      T* rstd_data = mean_data + batch_size;
      const T* input_data = input.data<T>();

      auto stream = cuda::get_cuda_stream();
      at::native::RowwiseMomentsCUDAKernel
        <<<batch_size, CUDA_BLOCK_REDUCE_NUM_THREADS, 0, stream>>>(
          depth, EPSILON, input_data, mean_data, rstd_data);
      at::native::LayerNormForwardCUDAKernel
        <<<batch_size, CUDA_NUM_THREADS, 0, stream>>>(
          depth, input_data, mean_data, rstd_data,
          gamma.data<T>(), beta.data<T>(), output.data<T>());
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    LayerNorm::compute<Device::CUDA, T>(const StorageView& beta,        \
                                        const StorageView& gamma,       \
                                        const StorageView& input,       \
                                        StorageView& output) const;

    DECLARE_IMPL(float)

  }
}

// The following CUDA kernels are adapted from:
// https://github.com/pytorch/pytorch/blob/295feb4e9af6cf4e7b9cff056de29a9dc17f50db/aten/src/ATen/native/cuda/layer_norm_kernel.cu
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
#  define WARP_SIZE 64
#else
#  define WARP_SIZE 32
#endif

    __inline__ __device__ float WarpReduceSum(float val) {
      #pragma unroll
      for (int offset = (WARP_SIZE >> 1); offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset, WARP_SIZE);
      }
      return val;
    }

    __inline__ __device__ float BlockReduceSum(float val, float* shared) {
      const int lid = threadIdx.x % WARP_SIZE;
      const int wid = threadIdx.x / WARP_SIZE;
      val = WarpReduceSum(val);
      if (lid == 0) {
        shared[wid] = val;
      }
      __syncthreads();
      val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lid] : 0;
      if (wid == 0) {
        val = WarpReduceSum(val);
      }
      return val;
    }

    __global__ void RowwiseMomentsCUDAKernel(int64_t N,
                                             float eps,
                                             const float* X,
                                             float* mean,
                                             float* rstd) {
      __shared__ float m_shared[WARP_SIZE];
      __shared__ float v_shared[WARP_SIZE];
      const int64_t i = blockIdx.x;
      float sum1 = 0;
      float sum2 = 0;
      for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
        const int64_t index = i * N + j;
        sum1 += X[index];
        sum2 += X[index] * X[index];
      }
      sum1 = BlockReduceSum(sum1, m_shared);
      sum2 = BlockReduceSum(sum2, v_shared);
      if (threadIdx.x == 0) {
        const float scale = float(1) / static_cast<float>(N);
        sum1 *= scale;
        sum2 = fmaxf(sum2 * scale - sum1 * sum1, float(0));
        mean[i] = sum1;
        rstd[i] = rsqrtf(sum2 + eps);
      }
    }

    __global__ void LayerNormForwardCUDAKernel(int64_t N,
                                               const float* X,
                                               const float* mean,
                                               const float* rstd,
                                               const float* gamma,
                                               const float* beta,
                                               float* Y) {
      const int64_t i = blockIdx.x;
      for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
        const int64_t index = i * N + j;
        const float gamma_v = gamma == nullptr ? float(1) : gamma[j];
        const float beta_v = beta == nullptr ? float(0) : beta[j];
        Y[index] = (X[index] - mean[i]) * rstd[i] * gamma_v + beta_v;
      }
    }

  }
}
