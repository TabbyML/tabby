#include "ctranslate2/ops/quantize.h"

#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {

    struct absolute_maximum_func {
      __device__ __forceinline__ float operator()(float a, float b) const {
        return fmaxf(fabsf(a), fabsf(b));
      }
    };

    struct maximum_func {
      __device__ __forceinline__ float operator()(float a, float b) const {
        return fmaxf(a, b);
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

    __global__ void quantize_kernel(const float* input,
                                    dim_t depth,
                                    float* scales,
                                    int8_t* output) {
      extern __shared__ unsigned char smem[];
      auto sdata = reinterpret_cast<float*>(smem);

      input += blockIdx.x * depth;
      output += blockIdx.x * depth;

      float thread_max = cuda::ilp_reduce(input, depth, absolute_maximum_func(), 0.f);
      float max = cuda::block_reduce(sdata, thread_max, maximum_func(), 0.f);
      float scale = 127.f / max;

      scales[blockIdx.x] = scale;

      cuda::apply_epilogue(input, depth, quantize_func(scale), output);
    }

    template<>
    void Quantize::quantize<Device::CUDA, int8_t>(const StorageView& input,
                                                  StorageView& output,
                                                  StorageView& scale) const {
      if (_shift_to_uint8)
        throw std::invalid_argument("Shift to uin8_t is not defined on CUDA");

      const dim_t batch_size = scale.size();
      const dim_t depth = input.dim(-1);

      const dim3 grid(batch_size);
      const dim3 block(cuda::get_block_size(depth));
      quantize_kernel<<<grid, block, block.x * sizeof (float), cuda::get_cuda_stream()>>>(
        input.data<float>(),
        depth,
        scale.data<float>(),
        output.data<int8_t>());
    }

  }
}
