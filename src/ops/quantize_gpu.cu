#include "ctranslate2/ops/quantize.h"

#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>

#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {

    struct absolute_maximum_func : public thrust::binary_function<float, float, float> {
      __host__ __device__
      float operator()(float a, float b) {
        return fmaxf(fabsf(a), fabsf(b));
      }
    };

    template <typename T>
    struct quantize_func : public thrust::binary_function<float, float, T> {
      __host__ __device__
      T operator()(float scale, float x) {
        return static_cast<T>(x * scale);
      }
    };

    template<>
    void Quantize::quantize<Device::CUDA, int8_t>(const StorageView& input,
                                                  StorageView& output,
                                                  StorageView& scale) const {
      if (_shift_to_uint8)
        throw std::invalid_argument("Shift to uin8_t is not defined on CUDA");

      const dim_t size = input.size();
      const dim_t batch_size = scale.size();
      const dim_t depth = input.dim(-1);

      const auto* input_data = input.data<float>();
      auto* output_data = output.data<int8_t>();
      auto* scale_data = scale.data<float>();

      // Assign 1 key per batch.
      auto keys_it = thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                                     cuda::repeat_vec_depth<int>(depth));

      // scales = 127.0 / reduce_max(abs(x), axis=1)
      THRUST_CALL(thrust::reduce_by_key,
                  keys_it, keys_it + size,
                  input_data,
                  thrust::make_discard_iterator(),
                  thrust::make_transform_output_iterator(
                    scale_data, static_cast<float>(127) / thrust::placeholders::_1),
                  thrust::equal_to<int>(),
                  absolute_maximum_func());

      // qx = x * expand_dims(scales, 1)
      cuda::binary_transform(scale_data, input_data, output_data, size,
                             quantize_func<int8_t>(),
                             cuda::repeat_vec_depth<dim_t>(depth));
    }

  }
}
