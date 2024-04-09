#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {
    class FlashAttention : public Op {
    public:
      FlashAttention(float queries_scale, dim_t sliding_window);

      void operator()(StorageView& queries,
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
                      dim_t offset) const;

    private:
      const float _queries_scale;
      const dim_t _sliding_window;
      template <Device D>
      void compute(StorageView& queries,
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
                   dim_t offset) const;
    };
  }
}
