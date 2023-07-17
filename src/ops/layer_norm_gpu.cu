#include "ctranslate2/ops/layer_norm.h"

#include "cuda/helpers.h"
#include "cuda/utils.h"

namespace at {
  namespace native {

    // Forward declaration of the CUDA kernels.
    template <typename T, typename SizeT>
    __global__ void LayerNormForwardCUDAKernel(SizeT N,
                                               float eps,
                                               const T* X,
                                               const T* gamma,
                                               const T* beta,
                                               T* Y);

  }
}

namespace ctranslate2 {
  namespace ops {

#define CUDA_NUM_THREADS 512

    template <Device D, typename T>
    void LayerNorm::compute(const StorageView* beta,
                            const StorageView* gamma,
                            const StorageView& input,
                            const dim_t axis,
                            const dim_t outer_size,
                            const dim_t axis_size,
                            const dim_t,
                            StorageView& output) const {
      if (axis != input.rank() - 1 || !beta || !gamma)
        throw std::invalid_argument("Generalized LayerNorm is currently not implemented on GPU");

      at::native::LayerNormForwardCUDAKernel<cuda::device_type<T>, cuda::index_t>
        <<<outer_size, CUDA_NUM_THREADS, 0, cuda::get_cuda_stream()>>>(
          axis_size,
          _epsilon,
          cuda::device_cast(input.data<T>()),
          cuda::device_cast(gamma->data<T>()),
          cuda::device_cast(beta->data<T>()),
          cuda::device_cast(output.data<T>()));
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    LayerNorm::compute<Device::CUDA, T>(const StorageView* beta,        \
                                        const StorageView* gamma,       \
                                        const StorageView& input,       \
                                        const dim_t axis,               \
                                        const dim_t outer_size,         \
                                        const dim_t axis_size,          \
                                        const dim_t inner_size,         \
                                        StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

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

#include <cub/block/block_reduce.cuh>

namespace at {
  namespace native {

    template <typename T, typename SizeT>
    __global__ void LayerNormForwardCUDAKernel(SizeT N,
                                               float eps,
                                               const T* X,
                                               const T* gamma,
                                               const T* beta,
                                               T* Y) {
      typedef cub::BlockReduce<float, CUDA_NUM_THREADS> BlockReduce;
      __shared__ typename BlockReduce::TempStorage m_temp_storage;
      __shared__ typename BlockReduce::TempStorage v_temp_storage;
      __shared__ float s_mean;
      __shared__ float s_variance;

      const SizeT i = blockIdx.x;

      float sum1 = 0;
      float sum2 = 0;
      for (SizeT j = threadIdx.x; j < N; j += blockDim.x) {
        const SizeT index = i * N + j;
        sum1 += float(X[index]);
        sum2 += float(X[index]) * float(X[index]);
      }
      sum1 = BlockReduce(m_temp_storage).Sum(sum1);
      sum2 = BlockReduce(v_temp_storage).Sum(sum2);
      if (threadIdx.x == 0) {
        const float scale = float(1) / float(N);
        sum1 *= scale;
        sum2 = fmaxf(sum2 * scale - sum1 * sum1, float(0));
        s_mean = sum1;
        s_variance = rsqrtf(sum2 + eps);
      }

      __syncthreads();

      for (SizeT j = threadIdx.x; j < N; j += blockDim.x) {
        const SizeT index = i * N + j;
        Y[index] = (float(X[index]) - s_mean) * s_variance * float(gamma[j]) + float(beta[j]);
      }
    }

  }
}
