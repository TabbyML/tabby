#include "ctranslate2/models/model_factory.h"

#include <mutex>

#include "ctranslate2/models/whisper.h"
#include "ctranslate2/models/transformer.h"

namespace ctranslate2 {
  namespace models {

    static void register_supported_models() {
      // Empty spec name, TransformerBase, and TransformerBig are there for backward compatibility.
      register_model<TransformerModel>("", /*num_heads=*/8);
      register_model<TransformerModel>("TransformerBase", /*num_heads=*/8);
      register_model<TransformerModel>("TransformerBig", /*num_heads=*/16);
      register_model<TransformerModel>("TransformerSpec");

      register_model<TransformerDecoderModel>("TransformerDecoderSpec");

      register_model<TransformerEncoderModel>("TransformerEncoderSpec");

      register_model<WhisperModel>("WhisperSpec");
    }

    std::shared_ptr<Model> create_model(const std::string& name) {
      static std::once_flag init_flag;
      std::call_once(init_flag, register_supported_models);

      return ModelFactory::get_instance().create_model(name);
    }

  }
}
