#include "ctranslate2/ops/topk.h"

#include "cuda/utils.h"
#include "cuda/helpers.h"

namespace fastertransformer {

#define MAX_BLOCKS_PER_BEAM 8

  template <typename T>
  void topK_kernelLauncher(const T* log_probs,
                           int* topk_tmp_id_buf,
                           T* topk_tmp_val_buf,
                           int* topk_id_buf,
                           T* topk_val_buf,
                           const int batch_size,
                           const int beams_per_batch,
                           const int k,
                           const int vocab_size,
                           cudaStream_t stream);

}

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename DataType, typename IndexType>
    void TopK::compute(const StorageView& x,
                       StorageView& values,
                       StorageView& indices) const {
      const dim_t depth = x.dim(-1);
      const dim_t batch_size = x.size() / depth;
      const dim_t temp_size = batch_size * _k * MAX_BLOCKS_PER_BEAM;

      auto& allocator = get_allocator<D>();
      auto* topk_tmp_id = static_cast<IndexType*>(
        allocator.allocate(temp_size * sizeof (IndexType)));
      auto* topk_tmp_val = static_cast<DataType*>(
        allocator.allocate(temp_size * sizeof (DataType)));

      fastertransformer::topK_kernelLauncher(cuda::device_cast(x.data<DataType>()),
                                             cuda::device_cast(topk_tmp_id),
                                             cuda::device_cast(topk_tmp_val),
                                             cuda::device_cast(indices.data<IndexType>()),
                                             cuda::device_cast(values.data<DataType>()),
                                             batch_size,
                                             1,
                                             _k,
                                             depth,
                                             cuda::get_cuda_stream());

      allocator.free(topk_tmp_id);
      allocator.free(topk_tmp_val);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    TopK::compute<Device::CUDA, T, int32_t>(const StorageView& x,       \
                                            StorageView& values,        \
                                            StorageView& indices) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}

// The kernels below were initially developed in FasterTransformer
// https://github.com/NVIDIA/DeepLearningExamples/blob/master/FasterTransformer/v3.0/fastertransformer/cuda/topk_kernels.cu
// which comes with the following license:

/*
  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

// We use an adaptation of these kernels that was proposed in MarianNMT
// https://github.com/rhenry-nv/marian-dev/blob/gpu_optimizations/src/3rd_party/topk.cuh
// which comes with the following license:


/*
  MIT License

  Copyright (c) 2016 Marcin Junczys-Dowmunt, the University of Edinburgh, Adam
  Mickiewicz University

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

#include <cub/block/block_reduce.cuh>

namespace fastertransformer {

#define NOT_FOUND -1

  template <typename T>
  __device__ __forceinline__ bool greater(const T& a, const T& b) {
    return a > b;
  }

#if !CUDA_CAN_USE_HALF
  template<>
  __device__ __forceinline__ bool greater(const __half& a, const __half& b) {
    return float(a) > float(b);
  }
#endif

  template <typename T>
  struct TopK {
    int p = NOT_FOUND;
    T u = cub::FpLimits<T>::Lowest();

    __device__ __forceinline__ void insert(T elem, int elem_id) {
      if (greater(elem, u)) {
        u = elem;
        p = elem_id;
      }
    }

    __device__ __forceinline__ void init() {
      u = cub::FpLimits<T>::Lowest();
      p = NOT_FOUND;
    }
  };

  template <typename T>
  __device__ __forceinline__ TopK<T>
  reduce_topk_op(const TopK<T>& a, const TopK<T>& b) {
    return greater(a.u, b.u) ? a : b;
  }

  template<typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
  __global__ void topk_stage_1(T* log_probs,
                               int* topk_tmp_id_buf,
                               T* topk_tmp_val_buf,
                               const int k,
                               const int vocab_size) {
    typedef cub::BlockReduce<TopK<T>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int row_id = bid / BLOCKS_PER_BEAM_; // row id for log_probs
    const int block_lane = bid % BLOCKS_PER_BEAM_; // block id for a beam
    const int tmp_log_buf_index = row_id * vocab_size;
    const int tmp_topk_buf_index = row_id * BLOCKS_PER_BEAM_ * k + block_lane * k;
    TopK<T> partial;

    for (int ite = 0; ite < k; ite++) {
      partial.init();
      #pragma unroll
      for (int elem_id = tid + block_lane * BLOCK_SIZE_;
           elem_id < vocab_size;
           elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_) {
        int index = elem_id + tmp_log_buf_index;
        partial.insert(log_probs[index], index);
      }

      TopK<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T>);

      if (tid == 0) {
        const int index = tmp_topk_buf_index + ite;
        topk_tmp_id_buf[index] = total.p;
        topk_tmp_val_buf[index] = total.u;
        // If we found a max, blank out the value in the log prob array before starting the next iteration
        if (total.p != NOT_FOUND)
          log_probs[total.p] = cub::FpLimits<T>::Lowest();
      }
      __syncthreads();
    }

    // Update prob array with original values.
    for (int beam = tid; beam < k; beam += BLOCK_SIZE_) {
      const int index = tmp_topk_buf_index + beam;
      int k_idx = topk_tmp_id_buf[index];
      if (k_idx != NOT_FOUND)
        log_probs[k_idx] = topk_tmp_val_buf[index];
    }
  }

  template<typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
  __global__ void topk_stage_2(const int* __restrict topk_tmp_id_buf,
                               T* topk_tmp_val_buf,
                               int* topk_id_buf,
                               T* topk_val_buf,
                               const int beams_per_batch,
                               const int vocab_size,
                               const int k) {

    const int size = beams_per_batch * k * BLOCKS_PER_BEAM_;
    const int tid = threadIdx.x;
    const int batch_id = blockIdx.x;

    typedef cub::BlockReduce<TopK<T>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    extern __shared__ char array[];
    T *s_val = topk_tmp_val_buf + batch_id * size;
    TopK<T> *topks = (TopK<T>*)(array);

    TopK<T> partial;

    for (int ite = 0; ite < k; ite++) {
      partial.init();
      #pragma unroll
      for (int i = tid; i < size; i+= BLOCK_SIZE_) {
        partial.insert(s_val[i], i);
      }

      TopK<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T>);

      if (tid == 0) {
        topks[ite] = total;
        s_val[total.p] = cub::FpLimits<T>::Lowest();
      }
      __syncthreads();
    }

    for (int beam = tid; beam < k; beam += BLOCK_SIZE_) {
      int indexInRow = topks[beam].p == NOT_FOUND? 0: topks[beam].p;
      int id = topk_tmp_id_buf[batch_id * size + indexInRow];
      id = id == NOT_FOUND? 0 : id; // If no max found, all values were equal to T::min so just return 0
      const int offset = batch_id * k + beam;
      topk_id_buf[offset] = id % vocab_size;
      topk_val_buf[offset] = topks[beam].u;
    }
  }

#define CASE_K(K,BLOCK_SIZE_1_, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_)        \
  case K:                                                               \
  topk_stage_1<T, BLOCK_SIZE_1_, BLOCKS_PER_BEAM_>                      \
  <<<batch_size * beams_per_batch * BLOCKS_PER_BEAM_, BLOCK_SIZE_1_, 0, stream>>>( \
    const_cast<T*>(log_probs),                                          \
    topk_tmp_id_buf,                                                    \
    topk_tmp_val_buf,                                                   \
    k, vocab_size);                                                     \
  topk_stage_2<T, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_>                      \
  <<<batch_size, BLOCK_SIZE_2_, K * sizeof(TopK<T>), stream>>>(         \
    topk_tmp_id_buf,                                                    \
    topk_tmp_val_buf,                                                   \
    topk_id_buf,                                                        \
    topk_val_buf,                                                       \
    beams_per_batch,                                                    \
    vocab_size,                                                         \
    k);                                                                 \
  break

  template <typename T>
  void topK_kernelLauncher(const T* log_probs,
                           int* topk_tmp_id_buf,
                           T* topk_tmp_val_buf,
                           int* topk_id_buf,
                           T* topk_val_buf,
                           const int batch_size,
                           const int beams_per_batch,
                           const int k,
                           const int vocab_size,
                           cudaStream_t stream) {
    switch (k) {
      CASE_K(1,128,128,MAX_BLOCKS_PER_BEAM);
      CASE_K(2,128,128,MAX_BLOCKS_PER_BEAM);
      CASE_K(4,128,128,MAX_BLOCKS_PER_BEAM);
      CASE_K(6,128,128,MAX_BLOCKS_PER_BEAM);
      CASE_K(8,128,128,MAX_BLOCKS_PER_BEAM);
      CASE_K(10,128,128,MAX_BLOCKS_PER_BEAM);
      CASE_K(16,128,128,5);
      CASE_K(32,256,128,1);
      CASE_K(64,256,256,1);
    default:
      topk_stage_1<T, 128, 1>
        <<<batch_size * beams_per_batch * 1, 128, 0, stream>>>(const_cast<T*>(log_probs),
                                                               topk_tmp_id_buf,
                                                               topk_tmp_val_buf,
                                                               k,
                                                               vocab_size);

      topk_stage_2<T, 128, 1>
        <<<batch_size, 128, k * sizeof(TopK<T>), stream>>>(topk_tmp_id_buf,
                                                           topk_tmp_val_buf,
                                                           topk_id_buf,
                                                           topk_val_buf,
                                                           beams_per_batch,
                                                           vocab_size,
                                                           k);
      break;
    }
  }

}
