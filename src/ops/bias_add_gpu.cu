#include "ctranslate2/ops/bias_add.h"

#include "type_dispatch.h"
#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {

    template <typename T, typename AddFunc, typename Epilogue>
    __global__ void bias_add_kernel(const T* value,
                                    const T* bias,
                                    T* output,
                                    dim_t depth,
                                    const AddFunc& add_func,
                                    const Epilogue& epilogue) {
      const dim_t i = blockIdx.x;
      for (dim_t j = threadIdx.x; j < depth; j += blockDim.x) {
        const dim_t index = i * depth + j;
        output[index] = epilogue(add_func(value[index], bias[j]));
      }
    }

    template <Device D, typename T>
    void BiasAdd::compute(const StorageView& value,
                          const StorageView& bias,
                          StorageView& output) const {
      const dim_t depth = bias.size();
      const dim_t batch_size = value.size() / depth;
      const dim_t blocks = std::min(batch_size, cuda::max_blocks);
      const dim_t threads = std::min(depth, cuda::max_threads);

      using DeviceT = cuda::device_type<T>;
      const auto* x = cuda::device_cast(value.data<T>());
      const auto* b = cuda::device_cast(bias.data<T>());
      auto* y = cuda::device_cast(output.data<T>());

      if (!_activation_type) {
        bias_add_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
          x, b, y, depth, cuda::plus<DeviceT>(), thrust::identity<DeviceT>());

      } else {
        switch (*_activation_type) {

        case ActivationType::ReLU:
          bias_add_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
            x, b, y, depth, cuda::plus<DeviceT>(), cuda::relu_func<DeviceT>());
          break;

        case ActivationType::GELU:
          bias_add_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
            x, b, y, depth, cuda::plus<DeviceT>(), cuda::gelu_func<DeviceT>());
          break;
        }
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    BiasAdd::compute<Device::CUDA, T>(const StorageView& value,         \
                                      const StorageView& bias,          \
                                      StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)

  }
}
