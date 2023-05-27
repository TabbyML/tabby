#include "ctranslate2-bindings/include/ctranslate2.h"

#include "ctranslate2/translator.h"
#include "ctranslate2/generator.h"

namespace tabby {
TextInferenceEngine::~TextInferenceEngine() {}

class EncoderDecoderImpl: public TextInferenceEngine {
 public:
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

  static std::unique_ptr<TextInferenceEngine> create(const ctranslate2::models::ModelLoader& loader) {
    auto impl = std::make_unique<EncoderDecoderImpl>();
    impl->translator_ = std::make_unique<ctranslate2::Translator>(loader);
    return impl;
  }
 private:
  std::unique_ptr<ctranslate2::Translator> translator_;
};

class DecoderImpl: public TextInferenceEngine {
 public:
  rust::Vec<rust::String> inference(
      rust::Slice<const rust::String> tokens,
      size_t max_decoding_length,
      float sampling_temperature,
      size_t beam_size
  ) const {
    // Create options.
    ctranslate2::GenerationOptions options;
    options.include_prompt_in_result = false;
    options.max_length = max_decoding_length;
    options.sampling_temperature = sampling_temperature;
    options.beam_size = beam_size;

    // Inference.
    std::vector<std::string> input_tokens(tokens.begin(), tokens.end());
    ctranslate2::GenerationResult result = generator_->generate_batch_async({ input_tokens }, options)[0].get();
    const auto& output_tokens = result.sequences[0];

    // Convert to rust vec.
    rust::Vec<rust::String> output;
    output.reserve(output_tokens.size());
    std::copy(output_tokens.begin(), output_tokens.end(), std::back_inserter(output));
    return output;
  }

  static std::unique_ptr<TextInferenceEngine> create(const ctranslate2::models::ModelLoader& loader) {
    auto impl = std::make_unique<DecoderImpl>();
    impl->generator_ = std::make_unique<ctranslate2::Generator>(loader);
    return impl;
  }
 private:
  std::unique_ptr<ctranslate2::Generator> generator_;
};

std::unique_ptr<TextInferenceEngine> create_engine(
    rust::Str model_path,
    rust::Str model_type,
    rust::Str device,
    rust::Slice<const int32_t> device_indices,
    size_t num_replicas_per_device
) {
  std::string model_type_str(model_type);
  std::string model_path_str(model_path);
  ctranslate2::models::ModelLoader loader(model_path_str);
  loader.device = ctranslate2::str_to_device(std::string(device));
  loader.device_indices = std::vector<int>(device_indices.begin(), device_indices.end());
  loader.num_replicas_per_device = num_replicas_per_device;

  if (model_type_str == "decoder") {
    return DecoderImpl::create(loader);
  } else if (model_type_str == "encoder-decoder") {
    return EncoderDecoderImpl::create(loader);
  } else {
    return nullptr;
  }
}
}  // namespace tabby
