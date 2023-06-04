#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Quantize : public Op {
    public:
      enum class ScaleType {
        GLOBAL,
        PER_LAYER,
        PER_ROW
      };

      static const float global_int16_scale;

      Quantize(const ScaleType int16_scale_type = ScaleType::GLOBAL,
               const bool shift_to_uint8 = false,
               const bool round_before_cast = true);

      void operator()(const StorageView& input,
                      StorageView& output,
                      StorageView& scale) const;

    private:
      template <Device D, typename InT, typename OutT>
      void quantize(const StorageView& input,
                    StorageView& output,
                    StorageView& scale) const;

      const ScaleType _int16_scale_type;
      const bool _shift_to_uint8;
      const bool _round_before_cast;
    };

  }
}
