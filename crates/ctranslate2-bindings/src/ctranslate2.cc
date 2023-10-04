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
  rust::Vec<uint32_t> inference(
      rust::Box<InferenceContext> context,
      InferenceCallback callback,
      rust::Slice<const rust::String> tokens,
      size_t max_decoding_length,
      float sampling_temperature
  ) const {
    // Inference.
    std::vector<std::string> input_tokens(tokens.begin(), tokens.end());
    return process(
        std::move(context),
        std::move(callback),
        input_tokens,
        Options{max_decoding_length, sampling_temperature}
    );
  }

  static std::unique_ptr<TextInferenceEngine> create(const ctranslate2::models::ModelLoader& loader) {
    auto impl = std::make_unique<Child>();
    impl->model_ = std::make_unique<Model>(loader);
    return impl;
  }

 protected:
  virtual rust::Vec<uint32_t> process(
      rust::Box<InferenceContext> context,
      InferenceCallback callback,
      const std::vector<std::string>& tokens,
      const Options& options) const = 0;
  std::unique_ptr<Model> model_;
};

class EncoderDecoderImpl : public TextInferenceEngineImpl<ctranslate2::Translator, EncoderDecoderImpl> {
 protected:
  virtual rust::Vec<uint32_t> process(
      rust::Box<InferenceContext> context,
      InferenceCallback callback,
      const std::vector<std::string>& tokens,
      const Options& options) const override {
    ctranslate2::TranslationOptions x;
    x.max_decoding_length = options.max_decoding_length;
    x.sampling_temperature = options.sampling_temperature;
    x.beam_size = 1;
    rust::Vec<uint32_t> output_ids;
    x.callback = [&](ctranslate2::GenerationStepResult result) {
      bool stop = callback(*context, result.step, result.token_id, result.token);
      if (!stop) {
        output_ids.push_back(result.token_id);
      } else if (result.is_last) {
        output_ids.push_back(result.token_id);
      }
      return stop;
    };
    ctranslate2::TranslationResult result = model_->translate_batch({ tokens }, x)[0];
    return output_ids;
  }
};

class DecoderImpl : public TextInferenceEngineImpl<ctranslate2::Generator, DecoderImpl> {
 protected:
  virtual rust::Vec<uint32_t> process(
      rust::Box<InferenceContext> context,
      InferenceCallback callback,
      const std::vector<std::string>& tokens,
      const Options& options) const override {
    ctranslate2::GenerationOptions x;
    x.include_prompt_in_result = false;
    x.max_length = options.max_decoding_length;
    x.sampling_temperature = options.sampling_temperature;
    x.beam_size = 1;

    rust::Vec<uint32_t> output_ids;
    x.callback = [&](ctranslate2::GenerationStepResult result) {
      bool stop = callback(*context, result.step, result.token_id, result.token);
      if (!stop) {
        output_ids.push_back(result.token_id);
      } else if (result.is_last) {
        output_ids.push_back(result.token_id);
      }
      return stop;
    };
    ctranslate2::GenerationResult result = model_->generate_batch_async({ tokens }, x)[0].get();
    return output_ids;
  }
};

std::shared_ptr<TextInferenceEngine> create_engine(
    rust::Str model_path,
    rust::Str model_type,
    rust::Str device,
    rust::Slice<const int32_t> device_indices
) {
  std::string model_type_str(model_type);
  std::string model_path_str(model_path);
  ctranslate2::models::ModelLoader loader(model_path_str);
  loader.device = ctranslate2::str_to_device(std::string(device));
  loader.device_indices = std::vector<int>(device_indices.begin(), device_indices.end());
  loader.compute_type = ctranslate2::ComputeType::AUTO;

  const size_t num_cpus = std::thread::hardware_concurrency();
  if (loader.device == ctranslate2::Device::CUDA) {
    // When device is cuda, set parallelism to be number of thread.
    loader.num_replicas_per_device = num_cpus;
  } else if (loader.device == ctranslate2::Device::CPU){
    // When device is cpu, adjust the number based on threads per replica.
    // https://github.com/OpenNMT/CTranslate2/blob/master/src/utils.cc#L77
    loader.num_replicas_per_device = std::max<int32_t>(num_cpus / 4, 1);
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
