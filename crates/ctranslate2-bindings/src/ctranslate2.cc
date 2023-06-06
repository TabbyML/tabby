#include "ctranslate2-bindings/include/ctranslate2.h"

#include "ctranslate2/translator.h"
#include "ctranslate2/generator.h"

namespace tabby {
TextInferenceEngine::~TextInferenceEngine() {}

template <class Model, class Child>
class TextInferenceEngineImpl : public TextInferenceEngine {
 protected:
  struct Options {
    size_t max_decoding_length;
    float sampling_temperature;
  };

 public:
  rust::Vec<rust::String> inference(
      rust::Box<InferenceContext> context,
      InferenceCallback callback,
      rust::Slice<const rust::String> tokens,
      size_t max_decoding_length,
      float sampling_temperature
  ) const {
    // Inference.
    std::vector<std::string> input_tokens(tokens.begin(), tokens.end());
    const auto output_tokens = process(
        std::move(context),
        std::move(callback),
        input_tokens,
        Options{max_decoding_length, sampling_temperature}
    );

    // Convert to rust vec.
    rust::Vec<rust::String> output;
    output.reserve(output_tokens.size());
    std::copy(output_tokens.begin(), output_tokens.end(), std::back_inserter(output));
    return output;
  }

  static std::unique_ptr<TextInferenceEngine> create(const ctranslate2::models::ModelLoader& loader) {
    auto impl = std::make_unique<Child>();
    impl->model_ = std::make_unique<Model>(loader);
    return impl;
  }

 protected:
  virtual std::vector<std::string> process(
      rust::Box<InferenceContext> context,
      InferenceCallback callback,
      const std::vector<std::string>& tokens,
      const Options& options) const = 0;
  std::unique_ptr<Model> model_;
};

class EncoderDecoderImpl : public TextInferenceEngineImpl<ctranslate2::Translator, EncoderDecoderImpl> {
 protected:
  virtual std::vector<std::string> process(
      rust::Box<InferenceContext> context,
      InferenceCallback callback,
      const std::vector<std::string>& tokens,
      const Options& options) const override {
    ctranslate2::TranslationOptions x;
    x.max_decoding_length = options.max_decoding_length;
    x.sampling_temperature = options.sampling_temperature;
    x.beam_size = 1;
    x.callback = [&](ctranslate2::GenerationStepResult result) {
      return callback(*context, result.step, result.token_id, result.token);
    };
    ctranslate2::TranslationResult result = model_->translate_batch({ tokens }, x)[0];
    return std::move(result.output());
  }
};

class DecoderImpl : public TextInferenceEngineImpl<ctranslate2::Generator, DecoderImpl> {
 protected:
  virtual std::vector<std::string> process(
      rust::Box<InferenceContext> context,
      InferenceCallback callback,
      const std::vector<std::string>& tokens,
      const Options& options) const override {
    ctranslate2::GenerationOptions x;
    x.include_prompt_in_result = false;
    x.max_length = options.max_decoding_length;
    x.sampling_temperature = options.sampling_temperature;
    x.beam_size = 1;
    x.callback = [&](ctranslate2::GenerationStepResult result) {
      return callback(*context, result.step, result.token_id, result.token);
    };
    ctranslate2::GenerationResult result = model_->generate_batch_async({ tokens }, x)[0].get();
    return std::move(result.sequences[0]);
  }
};

std::shared_ptr<TextInferenceEngine> create_engine(
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

  if (loader.device == ctranslate2::Device::CPU) {
    loader.compute_type = ctranslate2::ComputeType::INT8;
  } else if (loader.device == ctranslate2::Device::CUDA) {
    loader.compute_type = ctranslate2::ComputeType::FLOAT16;
  }

  if (model_type_str == "AutoModelForCausalLM") {
    return DecoderImpl::create(loader);
  } else if (model_type_str == "AutoModelForSeq2SeqLM") {
    return EncoderDecoderImpl::create(loader);
  } else {
    return nullptr;
  }
}
}  // namespace tabby
