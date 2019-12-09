#include "ctranslate2/ops/softmax.h"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/replace.h>

#include "../cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

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

      cudnnTensorDescriptor_t tensor_desc;
      CUDNN_CHECK(cudnnCreateTensorDescriptor(&tensor_desc));
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(tensor_desc,
                                             CUDNN_TENSOR_NCHW,
                                             cuda::TypeToCUDNNType<T>::value,
                                             batch_size,
                                             depth,
                                             1 /* h */, 1 /* w */));

      float alpha = 1;
      float beta = 0;
      CUDNN_CHECK(cudnnSoftmaxForward(cuda::get_cudnn_handle(),
                                      _log ? CUDNN_SOFTMAX_LOG : CUDNN_SOFTMAX_ACCURATE,
                                      CUDNN_SOFTMAX_MODE_INSTANCE,
                                      &alpha,
                                      tensor_desc,
                                      data,
                                      &beta,
                                      tensor_desc,
                                      output.data<float>()));
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(tensor_desc));
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    SoftMax::compute<Device::CUDA, T>(const StorageView& input,         \
                                      const StorageView* lengths,       \
                                      StorageView& output) const;

    DECLARE_IMPL(float)

  }
}
