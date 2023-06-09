#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class AlibiAdd : public BinaryOp {
    public:
      AlibiAdd(const bool use_positive_positions = false);

      void operator()(const StorageView& input,
                      const StorageView& alibi,
                      StorageView& output) const override;

    private:
      template <Device D, typename T>
      void compute(const StorageView& input,
                   const StorageView& alibi,
                   const dim_t alibi_offset,
                   StorageView& output) const;

      const bool _use_positive_positions;
    };

  }
}
