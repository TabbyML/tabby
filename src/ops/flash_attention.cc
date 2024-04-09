#include "ctranslate2/ops/flash_attention.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {
    FlashAttention::FlashAttention(float queries_scale, dim_t sliding_window)
    : _queries_scale(queries_scale)
    ,_sliding_window(sliding_window)
    {
    }

    void FlashAttention::operator()(StorageView& queries,
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
      DEVICE_DISPATCH(queries.device(), compute<D>(queries, keys, values, output, cached_keys, cached_values,
                                                   attention, return_normalized_attention,
                                                   rotary_cos, rotary_sin, rotary_interleave, alibi, offset));
    }
  }
}
