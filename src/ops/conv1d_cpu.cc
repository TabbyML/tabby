#include "ctranslate2/ops/conv1d.h"

#ifdef CT2_WITH_DNNL
#  include <dnnl.hpp>
#endif

namespace ctranslate2 {
  namespace ops {

    template<>
    void Conv1D::compute<Device::CPU, float>(const StorageView& input,
                                             const StorageView& weight,
                                             const StorageView* bias,
                                             StorageView& output) const {
#ifndef CT2_WITH_DNNL
      (void)input;
      (void)weight;
      (void)bias;
      (void)output;
      throw std::runtime_error("Conv1D on CPU currently requires the oneDNN library (a.k.a. DNNL) "
                               "which is not integrated in this build");

#else
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
#endif
    }

  }
}
