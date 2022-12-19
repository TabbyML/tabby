#include "ctranslate2/ops/conv1d.h"

#ifdef CT2_WITH_DNNL
#  include <dnnl.hpp>

namespace ctranslate2 {
  namespace ops {

    template<>
    void Conv1D::compute<Device::CPU, float>(const StorageView& input,
                                             const StorageView& weight,
                                             const StorageView* bias,
                                             StorageView& output) const {
      dnnl::engine engine(dnnl::engine::kind::cpu, 0);
      dnnl::stream engine_stream(engine);

      dnnl::memory::dims input_dims(input.shape().begin(), input.shape().end());
      dnnl::memory::dims output_dims(output.shape().begin(), output.shape().end());
      dnnl::memory::dims weight_dims(weight.shape().begin(), weight.shape().end());

      using tag = dnnl::memory::format_tag;
      using dt = dnnl::memory::data_type;

      dnnl::memory::desc input_md(input_dims, dt::f32, tag::any);
      dnnl::memory::desc output_md(output_dims, dt::f32, tag::any);
      dnnl::memory::desc weight_md(weight_dims, dt::f32, tag::any);

      dnnl::memory input_mem({input_dims, dt::f32, tag::ncw}, engine,
                             const_cast<void*>(input.buffer()));
      dnnl::memory output_mem({output_dims, dt::f32, tag::ncw}, engine,
                              output.buffer());
      dnnl::memory weight_mem({weight_dims, dt::f32, tag::oiw}, engine,
                              const_cast<void*>(weight.buffer()));

      dnnl::memory::dims stride{_stride};
      dnnl::memory::dims dilation{_dilation > 1 ? _dilation : 0};
      dnnl::memory::dims padding{_padding};

      std::unique_ptr<dnnl::convolution_forward::desc> conv_desc;
      std::unordered_map<int, dnnl::memory> args;
      args.reserve(4);

      if (bias) {
        dnnl::memory::dims bias_dims(bias->shape().begin(), bias->shape().end());
        dnnl::memory::desc bias_md(bias_dims, dt::f32, tag::a);
        dnnl::memory bias_mem(bias_md, engine, const_cast<void*>(bias->buffer()));
        args.emplace(DNNL_ARG_BIAS, bias_mem);

        conv_desc = std::make_unique<dnnl::convolution_forward::desc>(
          dnnl::prop_kind::forward_inference,
          dnnl::algorithm::convolution_direct,
          input_md,
          weight_md,
          bias_md,
          output_md,
          stride,
          dilation,
          padding,
          padding);

      } else {
        conv_desc = std::make_unique<dnnl::convolution_forward::desc>(
          dnnl::prop_kind::forward_inference,
          dnnl::algorithm::convolution_direct,
          input_md,
          weight_md,
          output_md,
          stride,
          dilation,
          padding,
          padding);
      }

      dnnl::convolution_forward::primitive_desc conv_pd(*conv_desc, engine);

      dnnl::memory conv_input_mem = input_mem;
      dnnl::memory conv_weight_mem = weight_mem;
      dnnl::memory conv_output_mem = output_mem;

      if (conv_pd.src_desc() != input_mem.get_desc()) {
        conv_input_mem = dnnl::memory(conv_pd.src_desc(), engine);
        dnnl::reorder(input_mem, conv_input_mem)
          .execute(engine_stream, input_mem, conv_input_mem);
      }

      if (conv_pd.weights_desc() != weight_mem.get_desc()) {
        conv_weight_mem = dnnl::memory(conv_pd.weights_desc(), engine);
        dnnl::reorder(weight_mem, conv_weight_mem)
          .execute(engine_stream, weight_mem, conv_weight_mem);
      }

      if (conv_pd.dst_desc() != output_mem.get_desc()) {
        conv_output_mem = dnnl::memory(conv_pd.dst_desc(), engine);
      }

      args.emplace(DNNL_ARG_SRC, conv_input_mem);
      args.emplace(DNNL_ARG_WEIGHTS, conv_weight_mem);
      args.emplace(DNNL_ARG_DST, conv_output_mem);

      dnnl::convolution_forward conv(conv_pd);
      conv.execute(engine_stream, args);

      if (conv_pd.dst_desc() != output_mem.get_desc()) {
        dnnl::reorder(conv_output_mem, output_mem)
          .execute(engine_stream, conv_output_mem, output_mem);
      }

      engine_stream.wait();
    }

  }
}

#else

#  ifdef CT2_WITH_MKL
#    include <mkl.h>
#  elif CT2_WITH_ACCELERATE
#    include <Accelerate/Accelerate.h>
#  elif CT2_WITH_OPENBLAS
#     include <cblas.h>
#  else
#     define CT2_NO_BLAS
#  endif

#  include "ctranslate2/ops/transpose.h"
#  include "cpu/parallel.h"

namespace ctranslate2 {
  namespace ops {

    static void conv1d_kernel(const float* input,
                              const float* weight,
                              const float* bias,
                              float* output,
                              dim_t batch_size,
                              dim_t input_length,
                              dim_t output_length,
                              dim_t in_channels,
                              dim_t out_channels,
                              dim_t kernel_size,
                              dim_t stride,
                              dim_t padding) {
      cpu::parallel_for(0, batch_size * out_channels, 1, [&](dim_t begin, dim_t end) {
        for (dim_t i = begin; i < end; ++i) {
          const dim_t b = i / out_channels;
          const dim_t c_out = i % out_channels;

          const float* filter = weight + (c_out * in_channels * kernel_size);
          const float* x = input + b * (in_channels * input_length);
          float* y = output + b * (out_channels * output_length);

          for (dim_t t_out = 0; t_out < output_length; ++t_out) {
            const dim_t t_in = t_out * stride - padding;

            const dim_t window_offset = std::clamp(t_in, dim_t(0), input_length);
            const dim_t window_end = std::clamp(t_in + kernel_size, dim_t(0), input_length);
            const dim_t window_size = window_end - window_offset;
            const dim_t filter_offset = window_offset - t_in;

            const float* window = x + (window_offset * in_channels);
            const float* kernel = filter + (filter_offset * in_channels);

#ifdef CT2_NO_BLAS
            float value = 0;
            for (dim_t j = 0; j < window_size * in_channels; ++j)
              value += window[j] * kernel[j];
#else
            float value = cblas_sdot(window_size * in_channels, window, 1, kernel, 1);
#endif

            if (bias)
              value += bias[c_out];

            y[c_out * output_length + t_out] = value;
          }
        }
      });
    }

    template<>
    void Conv1D::compute<Device::CPU, float>(const StorageView& input,
                                             const StorageView& weight,
                                             const StorageView* bias,
                                             StorageView& output) const {
      if (_dilation != 1)
        throw std::runtime_error("Dilation is not supported in this Conv1D implementation");

      const dim_t batch_size = input.dim(0);
      const dim_t in_channels = input.dim(1);
      const dim_t input_length = input.dim(2);
      const dim_t output_length = output.dim(2);
      const dim_t out_channels = weight.dim(0);
      const dim_t kernel_size = weight.dim(2);

      // Transpose input and weight to apply the kernel with a single contiguous dot.
      const Transpose transpose_op({0, 2, 1});
      StorageView input_t;
      StorageView weight_t;
      transpose_op(input, input_t);
      transpose_op(weight, weight_t);

      conv1d_kernel(input_t.data<float>(),
                    weight_t.data<float>(),
                    bias ? bias->data<float>() : nullptr,
                    output.data<float>(),
                    batch_size,
                    input_length,
                    output_length,
                    in_channels,
                    out_channels,
                    kernel_size,
                    _stride,
                    _padding);
    }

  }
}

#endif
