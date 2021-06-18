#include "ctranslate2/ops/activation.h"

#include "ctranslate2/ops/gelu.h"
#include "ctranslate2/ops/relu.h"

namespace ctranslate2 {
  namespace ops {

    const UnaryOp* get_activation(ActivationType type) {
      switch (type) {
      case ActivationType::ReLU: {
        static const ReLU relu;
        return &relu;
      }
      case ActivationType::GELU: {
        static const GELU gelu;
        return &gelu;
      }
      }
      return nullptr;
    }

  }
}
