#include "ctranslate2/ops/quantize.h"

namespace ctranslate2 {
  namespace ops {

    const StorageView Quantize::default_int16_scale(static_cast<float>(1000));

  }
}
