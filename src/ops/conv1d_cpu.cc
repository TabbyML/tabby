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

      std::unique_ptr<dnnl::convolution_forward::primitive_desc> conv_pd;
      std::unordered_map<int, dnnl::memory> args;
      args.reserve(4);

      if (bias) {
        dnnl::memory::dims bias_dims(bias->shape().begin(), bias->shape().end());
        dnnl::memory::desc bias_md(bias_dims, dt::f32, tag::a);
        dnnl::memory bias_mem(bias_md, engine, const_cast<void*>(bias->buffer()));
        args.emplace(DNNL_ARG_BIAS, bias_mem);

        conv_pd = std::make_unique<dnnl::convolution_forward::primitive_desc>(
          engine,
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
        conv_pd = std::make_unique<dnnl::convolution_forward::primitive_desc>(
          engine,
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

      dnnl::memory conv_input_mem = input_mem;
      dnnl::memory conv_weight_mem = weight_mem;
      dnnl::memory conv_output_mem = output_mem;

      if (conv_pd->src_desc() != input_mem.get_desc()) {
        conv_input_mem = dnnl::memory(conv_pd->src_desc(), engine);
        dnnl::reorder(input_mem, conv_input_mem)
          .execute(engine_stream, input_mem, conv_input_mem);
      }

      if (conv_pd->weights_desc() != weight_mem.get_desc()) {
        conv_weight_mem = dnnl::memory(conv_pd->weights_desc(), engine);
        dnnl::reorder(weight_mem, conv_weight_mem)
          .execute(engine_stream, weight_mem, conv_weight_mem);
      }

      if (conv_pd->dst_desc() != output_mem.get_desc()) {
        conv_output_mem = dnnl::memory(conv_pd->dst_desc(), engine);
      }

      args.emplace(DNNL_ARG_SRC, conv_input_mem);
      args.emplace(DNNL_ARG_WEIGHTS, conv_weight_mem);
      args.emplace(DNNL_ARG_DST, conv_output_mem);

      dnnl::convolution_forward conv(*conv_pd);
      conv.execute(engine_stream, args);

      if (conv_pd->dst_desc() != output_mem.get_desc()) {
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

#  include "ctranslate2/ops/gemm.h"
#  include "cpu/parallel.h"

namespace ctranslate2 {
  namespace ops {

    template<>
    void Conv1D::compute<Device::CPU, float>(const StorageView& input,
                                             const StorageView& weight,
                                             const StorageView* bias,
                                             StorageView& output) const {
      if (_dilation != 1)
        throw std::runtime_error("Dilation is not supported in this Conv1D implementation");

      compute_with_gemm(input, weight, output);
      // Add bias
      if (bias) {
        // Need to broadcast along dims 0 and 2, because output shape is:
        // batch_size, out_channels, output_length
        const auto batch_size = output.dim(0);
        const auto out_channels = output.dim(1);
        const auto output_length = output.dim(2);
        const auto a = bias->data<float>();
        const auto b = output.data<float>();
        cpu::parallel_for(0, batch_size * out_channels, 1, [&](dim_t begin, dim_t end){
          for (dim_t i = begin; i < end; ++i) {
            // Add bias element a_i to output_length elements at once
            // adjust index of `a` for items in the batch by calculating modulo
            const auto a_i = a[i % out_channels];
            const auto b_i = b + i * output_length;
            primitives<>::add(a_i, b_i, b_i, output_length);
          }
        });
      }
    }

    void Conv1D::compute_with_gemm(const StorageView& input,
                                   const StorageView& weight,
                                   StorageView& output) const {
      const dim_t batch_size = input.dim(0);
      const dim_t in_channels = input.dim(1);
      const dim_t out_channels = weight.dim(0);
      const dim_t kernel_size = weight.dim(2);
      const dim_t output_length = output.dim(2);

      std::vector im2col_output_shape{batch_size, in_channels * kernel_size, output_length};
      StorageView im2col_output(std::move(im2col_output_shape), static_cast<float>(0.0), Device::CPU);
      im2col(input, im2col_output, kernel_size);
      // Create a 2D view of weight to use in GEMM
      const StorageView weight_view({weight.dim(0), in_channels * kernel_size}, const_cast<float*>(weight.data<float>()));

      const dim_t m = out_channels;
      const dim_t n = output_length;
      const dim_t k = im2col_output.dim(1);
      const dim_t strideb = k * output_length;
      const dim_t stridec = out_channels * output_length;
      auto* b = im2col_output.data<float>();
      auto* c = output.data<float>();
      const Gemm gemm(1.0, 0.0, false, false);
      cpu::parallel_for(0, batch_size, 1, [&](dim_t begin, dim_t end) {
        for (dim_t i = begin; i < end; ++i) {
          float* b_i = b + (i * strideb);
          float* c_i = c + (i * stridec);
          StorageView cc({m, n}, c_i);
          StorageView bb({k, n}, b_i);
          gemm(weight_view, bb, cc);
        }
      });
    }

    void Conv1D::im2col(const StorageView& input, StorageView& output, const dim_t kernel_size) const {
      // input: batch_size x in_channels x input_length
      // output: batch_size x (in_channels * kernel_size) x output_length
      const dim_t batch_size = input.dim(0);
      const dim_t in_channels = input.dim(1);
      const dim_t input_length = input.dim(2);
      auto* out = output.data <float>();
      const auto* in = input.data <float>();
      dim_t input_channel_offset = 0;
      dim_t out_index = 0;
      for (int i = 0; i < batch_size; i++) {
        for (int c = 0; c < in_channels; c++) {
          // For each input channel fill (kernel_size * output_length) items in output array
          for (int k = 0; k < kernel_size; k++) {
            for (dim_t ti = -_padding; ti <= (input_length - kernel_size + _padding); ti += _stride) {
              // Fill items in [0, input_length) range
              const auto window_i = k + ti;
              if (0 <= window_i && window_i < input_length) {
                out[out_index] = in[window_i + input_channel_offset];
              }
              out_index += 1;
            }
          }
          input_channel_offset += input_length;
        }
      }
    }
  }
}

#endif
