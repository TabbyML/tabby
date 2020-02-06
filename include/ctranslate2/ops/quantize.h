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

      static const StorageView default_int16_scale;

      Quantize(ScaleType int16_scale_type = ScaleType::GLOBAL);
      void operator()(const std::vector<StorageView*>& inputs,
                      std::vector<StorageView*>& outputs) const override;
      void operator()(const StorageView& x,
                      StorageView& y,
                      StorageView& scale,
                      float shift = 0) const;

    private:
      ScaleType _int16_scale_type;
    };

  }
}
