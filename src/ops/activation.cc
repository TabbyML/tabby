#include "ctranslate2/ops/activation.h"

#include "ctranslate2/ops/gelu.h"
#include "ctranslate2/ops/relu.h"
#include "ctranslate2/ops/swish.h"

namespace ctranslate2 {
  namespace ops {

    const UnaryOp& get_activation_op(ActivationType type) {
      switch (type) {
      case ActivationType::ReLU: {
        static const ReLU relu;
        return relu;
      }
      case ActivationType::GELU: {
        static const GELU gelu;
        return gelu;
      }
      case ActivationType::Swish: {
        static const Swish swish;
        return swish;
      }
      }
      throw std::invalid_argument("invalid activation type");
    }

  }
}
