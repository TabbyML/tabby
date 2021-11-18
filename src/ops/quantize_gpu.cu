#include "ctranslate2/ops/quantize.h"

#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {

    template <typename T>
    struct absolute_maximum_func {
      __device__ __forceinline__ T operator()(T a, T b) const;
    };

    template<>
    struct absolute_maximum_func<float> {
      __device__ __forceinline__ float operator()(float a, float b) const {
        return fmaxf(fabsf(a), fabsf(b));
      }
    };

    template<>
    struct absolute_maximum_func<__half> {
      __device__ __forceinline__ __half operator()(__half a, __half b) const {
#if CUDA_CAN_USE_HALF && CUDA_VERSION >= 10020
        a = __habs(a);
        b = __habs(b);
        return a > b ? a : b;
#else
        return fmaxf(fabsf(a), fabsf(b));
#endif
      }
    };

    struct quantize_func {
      __device__ __forceinline__ quantize_func(float scale)
        : _scale(scale) {
      }

      __device__ __forceinline__ int8_t operator()(float v) const {
        return static_cast<int8_t>(v * _scale);
      }

    private:
      const float _scale;
    };

    template <typename T>
    __global__ void quantize_kernel(const T* input,
                                    cuda::index_t depth,
                                    float* scales,
                                    int8_t* output) {
      extern __shared__ unsigned char smem[];
      auto* sdata = reinterpret_cast<T*>(smem);

      input += blockIdx.x * depth;
      output += blockIdx.x * depth;

      T thread_max = cuda::ilp_reduce(input, depth, absolute_maximum_func<T>(), T(0));
      float max = cuda::block_reduce(sdata, thread_max, cuda::maximum<T>(), T(0));

      __shared__ float scale;

      if (threadIdx.x == 0) {
        scale = max != 0.f ? 127.f / max : 1.f;
        scales[blockIdx.x] = scale;
      }

      __syncthreads();

      cuda::apply_epilogue(input, depth, quantize_func(scale), output);
    }

    template <Device D, typename InT, typename OutT>
    void Quantize::quantize(const StorageView& input,
                            StorageView& output,
                            StorageView& scale) const {
      if (_shift_to_uint8)
        throw std::invalid_argument("Shift to uin8_t is not defined on CUDA");

      const dim_t batch_size = scale.size();
      const dim_t depth = input.dim(-1);

      const dim3 grid(batch_size);
      const dim3 block(cuda::get_block_size(depth));
      quantize_kernel<<<grid, block, block.x * sizeof (InT), cuda::get_cuda_stream()>>>(
        cuda::device_cast<InT>(input.data<InT>()),
        depth,
        scale.data<float>(),
        cuda::device_cast<OutT>(output.data<OutT>()));
    }

    template void
    Quantize::quantize<Device::CUDA, float, int8_t>(const StorageView&,
                                                    StorageView&,
                                                    StorageView&) const;
    template void
    Quantize::quantize<Device::CUDA, float16_t, int8_t>(const StorageView&,
                                                        StorageView&,
                                                        StorageView&) const;

  }
}
