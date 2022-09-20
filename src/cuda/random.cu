#include "random.h"

#include <memory>

#include "ctranslate2/allocator.h"
#include "ctranslate2/random.h"
#include "ctranslate2/utils.h"
#include "utils.h"

namespace ctranslate2 {
  namespace cuda {

    template <typename curandState>
    __global__ void init_curand_states_kernel(curandState* states, unsigned long long seed) {
      const auto id = threadIdx.x + blockIdx.x * blockDim.x;
      curand_init(seed, id, 0, states + id);
    }

    template <typename curandState>
    class ScopedCurandStates {
    public:
      ScopedCurandStates(size_t num_states)
        : _allocator(get_allocator<Device::CUDA>())
      {
        constexpr size_t num_init_threads = 32;
        const size_t blocks = ceil_divide(num_states, num_init_threads);
        _num_states = blocks * num_init_threads;
        _states = static_cast<curandState*>(_allocator.allocate(_num_states * sizeof (curandState)));
        init_curand_states_kernel<<<blocks, num_init_threads, 0, cuda::get_cuda_stream()>>>(
          _states, get_random_seed());
      }

      ~ScopedCurandStates() {
        _allocator.free(_states);
      }

      size_t num_states() const {
        return _num_states;
      }

      curandState* states() {
        return _states;
      }

    private:
      Allocator& _allocator;
      size_t _num_states;
      curandState* _states;
    };

    curandStatePhilox4_32_10_t* get_curand_states(size_t num_states) {
      static thread_local std::unique_ptr<ScopedCurandStates<curandStatePhilox4_32_10_t>> states;
      if (!states || num_states > states->num_states())
        states = std::make_unique<ScopedCurandStates<curandStatePhilox4_32_10_t>>(num_states);
      return states->states();
    }

  }
}
