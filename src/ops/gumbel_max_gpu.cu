#include "ctranslate2/ops/gumbel_max.h"

#include "type_dispatch.h"
#include "cuda/helpers.h"
#include "cuda/random.h"

namespace ctranslate2 {
  namespace ops {

    class add_gumbel_noise_func {
    public:
      add_gumbel_noise_func(curandStatePhilox4_32_10_t* states)
        : _states(states)
      {
      }

      template <typename DataType, typename IndexType>
      __device__ DataType operator()(DataType value, IndexType id) const {
        const float z = -logf(curand_uniform(_states + id));
        return float(value) + z;
      }

    private:
      curandStatePhilox4_32_10_t* _states;
    };

    template <Device D, typename T>
    void GumbelMax::add_gumbel_noise(const StorageView& x, StorageView& y) const {
      THRUST_CALL(thrust::transform,
                  cuda::device_cast(x.data<T>()),
                  cuda::device_cast(x.data<T>()) + x.size(),
                  thrust::counting_iterator<cuda::index_t>(0),
                  cuda::device_cast(y.data<T>()),
                  add_gumbel_noise_func(cuda::get_curand_states(x.size())));
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    GumbelMax::add_gumbel_noise<Device::CUDA, T>(const StorageView& x,  \
                                                 StorageView& y) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
