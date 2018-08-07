#include "ctranslate2/ops/softmax.h"
#include "ctranslate2/ops/log_softmax.h"

#include "ctranslate2/cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

    static void cudnn_softmax(const StorageView& input,
                              StorageView& output,
                              cudnnSoftmaxAlgorithm_t algorithm) {
      static thread_local cudnnTensorDescriptor_t tensor_desc;
      static thread_local bool tensor_desc_init = false;

      if (!tensor_desc_init) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&tensor_desc));
        tensor_desc_init = true;
      }

      size_t depth = input.dim(-1);
      size_t batch_size = input.size() / depth;
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(tensor_desc,
                                             CUDNN_TENSOR_NCHW,
                                             CUDNN_DATA_FLOAT,
                                             batch_size,
                                             depth,
                                             1 /* h */, 1 /* w */));

      float alpha = 1;
      float beta = 0;
      CUDNN_CHECK(cudnnSoftmaxForward(cuda::get_cudnn_handle(),
                                      algorithm,
                                      CUDNN_SOFTMAX_MODE_INSTANCE,
                                      &alpha,
                                      tensor_desc,
                                      input.data<float>(),
                                      &beta,
                                      tensor_desc,
                                      output.data<float>()));
    }

    template <Device D, typename T>
    void SoftMax::compute(const StorageView& input, StorageView& output) const {
      cudnn_softmax(input, output, CUDNN_SOFTMAX_ACCURATE);
    }

    template <Device D, typename T>
    void LogSoftMax::compute(const StorageView& input, StorageView& output) const {
      cudnn_softmax(input, output, CUDNN_SOFTMAX_LOG);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    SoftMax::compute<Device::CUDA, T>(const StorageView& input,         \
                                      StorageView& output) const;       \
    template void                                                       \
    LogSoftMax::compute<Device::CUDA, T>(const StorageView& input,      \
                                         StorageView& output) const;

    DECLARE_IMPL(float)

  }
}
