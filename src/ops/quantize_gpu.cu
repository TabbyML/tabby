#include "ctranslate2/ops/quantize.h"

#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {

    template <typename T>
    struct absolute_maximum_func {
      __device__ __forceinline__ T operator()(T a, T b) const {
        return fmaxf(fabsf(a), fabsf(b));
      }
    };

#if CUDA_CAN_USE_HALF
    template<>
    struct absolute_maximum_func<__half> {
      __device__ __forceinline__ __half operator()(__half a, __half b) const {
        a = __habs(a);
        b = __habs(b);
        return a > b ? a : b;
      }
    };
#endif

#if CUDA_CAN_USE_BF16_MATH
    template<>
    struct absolute_maximum_func<__nv_bfloat16> {
      __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const {
        return __hmax(__habs(a), __habs(b));
      }
    };
#endif

    struct rescale_func {
      __device__ __forceinline__ rescale_func(float scale)
        : _scale(scale) {
      }
      __device__ __forceinline__ float operator()(float v) const {
        return v * _scale;
      }
    private:
      const float _scale;
    };

    struct rescale_and_round_func {
      __device__ __forceinline__ rescale_and_round_func(float scale)
        : _scale(scale) {
      }
      __device__ __forceinline__ float operator()(float v) const {
        return nearbyintf(v * _scale);
      }
    private:
      const float _scale;
    };

    template <typename T>
    __global__ void quantize_kernel(const T* input,
                                    cuda::index_t depth,
                                    float* scales,
                                    int8_t* output,
                                    bool round_before_cast) {
      extern __shared__ unsigned char smem[];
      auto* sdata = reinterpret_cast<T*>(smem);

      input += blockIdx.x * depth;
      output += blockIdx.x * depth;

      T thread_max = cuda::ilp_reduce(input, depth, absolute_maximum_func<T>(), T(0.f));
      float max = cuda::block_reduce(sdata, thread_max, cuda::maximum<T>(), T(0.f));

      __shared__ float scale;

      if (threadIdx.x == 0) {
        scale = max != 0.f ? 127.f / max : 1.f;
        scales[blockIdx.x] = scale;
      }

      __syncthreads();

      if (round_before_cast)
        cuda::apply_epilogue(input, depth, rescale_and_round_func(scale), output);
      else
        cuda::apply_epilogue(input, depth, rescale_func(scale), output);
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
        cuda::device_cast<OutT>(output.data<OutT>()),
        _round_before_cast);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Quantize::quantize<Device::CUDA, T, int8_t>(const StorageView&,     \
                                                StorageView&,           \
                                                StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
