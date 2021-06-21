#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    // This enum order should remain fixed.
    enum class ActivationType {
      ReLU,
      GELU,
    };

    const UnaryOp& get_activation_op(ActivationType type);

  }
}
