#include "ctranslate2-bindings/include/ctranslate2.h"

#include "ctranslate2/translator.h"

namespace tabby {
TextInferenceEngine::~TextInferenceEngine() {}

class TextInferenceEngineImpl : public TextInferenceEngine {
 public:
  TextInferenceEngineImpl(const std::string& model_path) {
    ctranslate2::models::ModelLoader loader(model_path);
    translator_ = std::make_unique<ctranslate2::Translator>(loader);
  }

  ~TextInferenceEngineImpl() {}

  rust::Vec<rust::String> inference(
      rust::Slice<const rust::String> tokens,
      size_t max_decoding_length,
      float sampling_temperature,
      size_t beam_size
  ) const {
    // Create options.
    ctranslate2::TranslationOptions options;
    options.max_decoding_length = max_decoding_length;
    options.sampling_temperature = sampling_temperature;
    options.beam_size = beam_size;

    // Inference.
    std::vector<std::string> input_tokens(tokens.begin(), tokens.end());
    ctranslate2::TranslationResult result = translator_->translate_batch({ input_tokens }, options)[0];
    const auto& output_tokens = result.output();

    // Convert to rust vec.
    rust::Vec<rust::String> output;
    output.reserve(output_tokens.size());
    std::copy(output_tokens.begin(), output_tokens.end(), std::back_inserter(output));
    return output;
  }
 private:
  std::unique_ptr<ctranslate2::Translator> translator_;
};

std::unique_ptr<TextInferenceEngine> create_engine(rust::Str model_path) {
  return std::make_unique<TextInferenceEngineImpl>(std::string(model_path));
}
}  // namespace tabby
