#include "ctranslate2/ops/flash_attention.h"
#include "ctranslate2/ops/flash-attention/flash.h"
#include "ctranslate2/ops/flash-attention/static_switch.h"
#include "ctranslate2/ops/transpose.h"
#include "ctranslate2/ops/slide.h"
#include "cuda/utils.h"

#include "dispatch.h"

#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634074
#endif

namespace ctranslate2 {
  namespace ops {
    static void set_params_fprop(Flash_fwd_params &params,
      // sizes
                                 const size_t b,
                                 const size_t seqlen_q,
                                 const size_t seqlen_k,
                                 const size_t seqlen_q_rounded,
                                 const size_t seqlen_k_rounded,
                                 const size_t h,
                                 const size_t h_k,
                                 const size_t d,
                                 const size_t d_rounded,
      // device pointers
                                 StorageView* q,
                                 StorageView* k,
                                 StorageView* v,
                                 StorageView* out,
                                 void *cu_seqlens_q_d,
                                 void *cu_seqlens_k_d,
                                 void *seqused_k,
                                 void *p_d,
                                 void *softmax_lse_d,
                                 float softmax_scale,
                                 int window_size_left,
                                 int window_size_right,
                                 bool seqlenq_ngroups_swapped=false) {

      // Reset the parameters
      memset(&params, 0, sizeof(params));

      params.is_bf16 = q->dtype() == DataType::BFLOAT16;

      // Set the pointers and strides.
      params.q_ptr = q->buffer();
      params.k_ptr = k->buffer();
      params.v_ptr = v->buffer();
      // All stride are in elements, not bytes.
      params.q_row_stride = q->stride(-3);
      params.k_row_stride = k->stride(-3);
      params.v_row_stride = v->stride(-3);
      params.q_head_stride = q->stride(-2);
      params.k_head_stride = k->stride(-2);
      params.v_head_stride = v->stride(-2);
      params.o_ptr = out->buffer();
      params.o_row_stride = out->stride(-3);
      params.o_head_stride = out->stride(-2);

      if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q->stride(0);
        params.k_batch_stride = k->stride(0);
        params.v_batch_stride = v->stride(0);
        params.o_batch_stride = out->stride(0);
        if (seqlenq_ngroups_swapped) {
          params.q_batch_stride *= seqlen_q;
          params.o_batch_stride *= seqlen_q;
        }
      }

      params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
      params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
      params.seqused_k = static_cast<int *>(seqused_k);

      // P = softmax(QK^T)
      params.p_ptr = p_d;

      // Softmax sum
      params.softmax_lse_ptr = softmax_lse_d;

      // Set the dimensions.
      params.b = b;
      params.h = h;
      params.h_k = h_k;
      params.h_h_k_ratio = h / h_k;
      params.seqlen_q = seqlen_q;
      params.seqlen_k = seqlen_k;
      params.seqlen_q_rounded = seqlen_q_rounded;
      params.seqlen_k_rounded = seqlen_k_rounded;
      params.d = d;
      params.d_rounded = d_rounded;

      // Set the different scale values.
      params.scale_softmax = softmax_scale;
      params.scale_softmax_log2 = softmax_scale * M_LOG2E;

      // Set this to probability of keeping an element to simplify things.
      // not use dropout
      params.p_dropout = 1.f;
      params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
      params.rp_dropout = 1.f / params.p_dropout;
      params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;

      // Causal is the special case where window_size_right == 0 and window_size_left < 0.
      // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
      params.is_causal = window_size_left < 0 && window_size_right == 0;

      if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
      if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
      params.window_size_left = window_size_left;
      params.window_size_right = window_size_right;

      params.is_seqlens_k_cumulative = true;
    }

    // Find the number of splits that maximizes the occupancy. For example, if we have
    // batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
    // better than having 3 splits (efficiency = 0.67). However, we also don't want too many
    // splits as that would incur more HBM reads/writes.
    // So we find the best efficiency, then find the smallest number of splits that gets 85%
    // of the best efficiency.
    static int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
      // If we have enough to almost fill the SMs, then just use 1 split
      if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
      max_splits = std::min({max_splits, num_SMs, num_n_blocks});
      float max_efficiency = 0.f;
      std::vector<float> efficiency;
      efficiency.reserve(max_splits);
      auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
      // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
      // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
      // (i.e. it's 11 splits anyway).
      // So we check if the number of blocks per split is the same as the previous num_splits.
      auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
      };
      for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
          efficiency.push_back(0.f);
        } else {
          float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
          float eff = n_waves / ceil(n_waves);
          // printf("num_splits = %d, eff = %f\n", num_splits, eff);
          if (eff > max_efficiency) { max_efficiency = eff; }
          efficiency.push_back(eff);
        }
      }
      for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
          // printf("num_splits chosen = %d\n", num_splits);
          return num_splits;
        }
      }
      return 1;
    }

    static void set_params_splitkv(Flash_fwd_params &params, const int batch_size,
                                   const int num_heads, const int head_size, const int max_seqlen_k, const int max_seqlen_q,
                                   const int head_size_rounded,
                                   const int num_splits, cudaDeviceProp *dprops) {

      // This needs to match with run_mha_fwd_splitkv_dispatch
      const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
      const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
      // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
      // In any case we don't expect seqlen_q to be larger than 64 for inference.
      const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
      params.num_splits = num_splits;
      if (num_splits < 1) {
        params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks, dprops->multiProcessorCount, num_n_blocks, 128);
      }
      TENSOR_CHECK((params.num_splits <= 128), "[FlashAttention] num_splits > 128 not supported");
    }

    void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
      FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
          if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
            run_mha_fwd_<elem_type, kHeadDim>(params, stream);
          } else {
            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim>(params, stream);
          }
        });
      });
    }

    static const ops::Transpose transpose_op({0, 2, 1, 3});

    template<>
    void FlashAttention::compute<Device::CUDA>(StorageView& queries,
                                               StorageView& keys,
                                               StorageView& values,
                                               StorageView& output,
                                               StorageView* cached_keys,
                                               StorageView* cached_values,
                                               StorageView* attention,
                                               bool return_normalized_attention,
                                               StorageView* rotary_cos,
                                               StorageView* rotary_sin,
                                               const bool rotary_interleave,
                                               StorageView* alibi,
                                               dim_t offset) const {
      const Device device = queries.device();
      const DataType dtype = queries.dtype();
      StorageView rotary_cos_half(dtype, device);
      StorageView rotary_sin_half(dtype, device);

      dim_t window_size_left = _sliding_window > 0 ? _sliding_window : -1;
      dim_t window_size_right = _sliding_window > 0 ? 0 : -1;

      int device_id = ctranslate2::get_device_index(ctranslate2::Device::CUDA);
      auto dprops = ctranslate2::cuda::get_device_properties(device_id);

      const auto shape = queries.shape();
      const dim_t batch_size = shape[0];
      dim_t seqlen_q = shape[1];
      dim_t num_heads = shape[2];
      const dim_t head_size_og = shape[3];

      dim_t seqlen_k, num_heads_k;
      if (offset == 0) {
        seqlen_k = keys.dim(1);
        num_heads_k = keys.dim(2);
        if (window_size_left >= seqlen_k) { window_size_left = -1; }
        if (window_size_right >= seqlen_k) { window_size_right = -1; }
      } else {
        seqlen_k = cached_keys->dim(1);
        num_heads_k = cached_keys->dim(2);
      }

      // causal=true is the same as causal=false in this case
      bool is_causal = true;
      if (seqlen_q == 1 && !alibi) { is_causal = false; }
      if (is_causal) { window_size_right = 0; }

      // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
      // H/t Daniel Haziza
      const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && head_size_og % 8 == 0;
      if (seqlenq_ngroups_swapped) {
        const int ngroups = num_heads / num_heads_k;
        StorageView tmp(dtype, device);
        transpose_op(queries.reshape({batch_size, num_heads_k, ngroups, head_size_og}), tmp);
        queries = std::move(tmp);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
      }

      if (offset > 0) {
        if (window_size_left >= seqlen_k) { window_size_left = -1; }
        if (window_size_right >= seqlen_k) { window_size_right = -1; }
      }

      auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
      const int head_size = round_multiple(head_size_og, 8);
      const int head_size_rounded = round_multiple(head_size, 32);
      const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
      const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

      StorageView softmax_lse({batch_size, num_heads, seqlen_q}, DataType::FLOAT32, device);
      output.resize(queries.shape());
      if (attention && return_normalized_attention) {
        attention->resize({batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded});
      }
      bool force_split_kernel = false;
      StorageView seqlens_k({batch_size}, static_cast<int>(offset), device);

      Flash_fwd_params params;
      if (offset == 0) {
        set_params_fprop(params,
                         batch_size,
                         seqlen_q, seqlen_k,
                         seqlen_q_rounded, seqlen_k_rounded,
                         num_heads, num_heads_k,
                         head_size, head_size_rounded,
                         &queries, &keys, &values, &output,
                        /*cu_seqlens_q_d=*/nullptr,
                        /*cu_seqlens_k_d=*/nullptr,
                        /*seqused_k=*/nullptr,
                         (return_normalized_attention && attention) ? attention->buffer() : /*p_ptr=*/nullptr,
                         softmax_lse.buffer(),
                         _queries_scale,
                         window_size_left,
                         window_size_right);

        // set params splitkv
        set_params_splitkv(params, batch_size, num_heads,
                           head_size, seqlen_k, seqlen_q,
                           head_size_rounded, /*num_splits*/0, &dprops);
      }
      else {
        const int page_block_size = 1;

        set_params_fprop(params,
                         batch_size,
                         seqlen_q, seqlen_k,
                         seqlen_q_rounded, seqlen_k_rounded,
                         num_heads, num_heads_k,
                         head_size, head_size_rounded,
                         &queries, cached_keys, cached_values, &output,
                         /*cu_seqlens_q_d=*/nullptr,
                         /*cu_seqlens_k_d=*/nullptr,
                         /*seqused_k=*/nullptr,
                         /*p_ptr=*/nullptr,
                         softmax_lse.buffer(),
                         _queries_scale,
                         window_size_left,
                         window_size_right);

        int seqlen_knew = keys.dim(1);
        params.seqlen_knew = seqlen_knew;
        params.knew_ptr = keys.buffer();
        params.vnew_ptr = values.buffer();
        // All stride are in elements, not bytes.
        params.knew_batch_stride = keys.stride(0);
        params.vnew_batch_stride = values.stride(0);
        params.knew_row_stride = keys.stride(-3);
        params.vnew_row_stride = values.stride(-3);
        params.knew_head_stride = keys.stride(-2);
        params.vnew_head_stride = values.stride(-2);
        params.cu_seqlens_k =  static_cast<int *>(seqlens_k.buffer());
        params.is_seqlens_k_cumulative = false;

        if (rotary_cos && rotary_sin) {
          params.rotary_dim = rotary_cos->dim(1);
          const ops::Slide slide_op(1, 0, params.rotary_dim / 2);
          slide_op(*rotary_cos, rotary_cos_half);
          slide_op(*rotary_sin, rotary_sin_half);
          params.rotary_cos_ptr = rotary_cos_half.buffer();
          params.rotary_sin_ptr = rotary_sin_half.buffer();
          params.is_rotary_interleaved = rotary_interleave;
        }
        else
          params.rotary_dim = 0;

        set_params_splitkv(params, batch_size, num_heads,
                           head_size, seqlen_k, seqlen_q,
                           head_size_rounded, /*num_splits*/0, &dprops);
        params.page_block_size = page_block_size;
        force_split_kernel = true;
      }

      StorageView softmax_lse_accum(DataType::FLOAT32, device);
      StorageView out_accum(DataType::FLOAT32, device);
      if (params.num_splits > 1) {
        softmax_lse_accum.resize({params.num_splits, batch_size, num_heads, seqlen_q});
        out_accum.resize({params.num_splits, batch_size, num_heads, seqlen_q, head_size_rounded});
        params.softmax_lseaccum_ptr = softmax_lse_accum.buffer();
        params.oaccum_ptr = out_accum.buffer();
      }
      params.alibi_slopes_ptr = nullptr;

      cudaStream_t stream = ctranslate2::cuda::get_cuda_stream();
      run_mha_fwd(params, stream, force_split_kernel);

      if (seqlenq_ngroups_swapped) {
        StorageView tmp(dtype, device);
        transpose_op(output, tmp);
        output = std::move(tmp);
        output.reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og});
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
      }
    }
  }
}
