#include "ctranslate2/translator.h"

#include "ctranslate2/batch_reader.h"

namespace ctranslate2 {

  Translator::Translator(const std::string& model_dir,
                         Device device,
                         int device_index,
                         ComputeType compute_type) {
    set_model(models::Model::load(model_dir, device, device_index, compute_type));
  }

  Translator::Translator(models::ModelReader& model_reader,
                         Device device,
                         int device_index,
                         ComputeType compute_type)
  {
    set_model(models::Model::load(model_reader, device, device_index, compute_type));
  }

  Translator::Translator(const std::shared_ptr<const models::Model>& model) {
    set_model(model);
  }

  Translator::Translator(const Translator& other) {
    if (other._model)
      set_model(other._model);
  }

  TranslationResult
  Translator::translate(const std::vector<std::string>& tokens) {
    return translate(tokens, TranslationOptions());
  }

  TranslationResult
  Translator::translate(const std::vector<std::string>& tokens,
                        const TranslationOptions& options) {
    return translate_batch({tokens}, options)[0];
  }

  TranslationResult
  Translator::translate_with_prefix(const std::vector<std::string>& source,
                                    const std::vector<std::string>& target_prefix,
                                    const TranslationOptions& options) {
    return translate_batch_with_prefix({source}, {target_prefix}, options)[0];
  }

  std::vector<TranslationResult>
  Translator::translate_batch(const std::vector<std::vector<std::string>>& batch_tokens) {
    return translate_batch(batch_tokens, TranslationOptions());
  }

  std::vector<TranslationResult>
  Translator::translate_batch(const std::vector<std::vector<std::string>>& batch_tokens,
                              const TranslationOptions& options) {
    return translate_batch_with_prefix(batch_tokens, {}, options);
  }

  std::vector<TranslationResult>
  Translator::translate_batch_with_prefix(const std::vector<std::vector<std::string>>& source,
                                          const std::vector<std::vector<std::string>>& target_prefix,
                                          const TranslationOptions& options) {
    assert_has_model();
    register_current_allocator();

    options.validate();
    if (source.empty())
      return {};

    const TranslationResult empty_result(options.num_hypotheses,
                                         options.return_attention,
                                         options.return_scores);
    std::vector<TranslationResult> results(source.size(), empty_result);

    for (const auto& batch : rebatch_input(load_examples({source, target_prefix}), 0)) {
      auto batch_results = _model->sample(*_encoder,
                                          *_decoder,
                                          batch.get_stream(0),
                                          batch.get_stream(1),
                                          *options.make_search_strategy(),
                                          *options.make_sampler(),
                                          options.use_vmap,
                                          options.max_input_length,
                                          options.max_decoding_length,
                                          options.min_decoding_length,
                                          options.num_hypotheses,
                                          options.return_alternatives,
                                          options.return_scores,
                                          options.return_attention,
                                          options.replace_unknowns,
                                          options.normalize_scores,
                                          options.repetition_penalty,
                                          options.disable_unk);

      for (size_t i = 0; i < batch_results.size(); ++i)
        results[batch.example_index[i]] = std::move(batch_results[i]);
    }

    return results;
  }

  std::vector<ScoringResult>
  Translator::score_batch(const std::vector<std::vector<std::string>>& source,
                          const std::vector<std::vector<std::string>>& target,
                          const ScoringOptions& options) {
    assert_has_model();
    register_current_allocator();
    if (source.empty())
      return {};
    return _model->score(*_encoder, *_decoder, source, target, options.max_input_length);
  }

  Device Translator::device() const {
    assert_has_model();
    return _model->device();
  }

  int Translator::device_index() const {
    assert_has_model();
    return _model->device_index();
  }

  ComputeType Translator::compute_type() const {
    assert_has_model();
    return _model->compute_type();
  }

  void Translator::set_model(const std::string& model_dir) {
    models::ModelFileReader model_reader(model_dir);
    set_model(model_reader);
  }

  void Translator::set_model(models::ModelReader& model_reader) {
    Device device = Device::CPU;
    int device_index = 0;
    ComputeType compute_type = ComputeType::DEFAULT;
    if (_model) {
      device = _model->device();
      device_index = _model->device_index();
      compute_type = _model->compute_type();
    }
    set_model(models::Model::load(model_reader, device, device_index, compute_type));
  }

  void Translator::set_model(const std::shared_ptr<const models::Model>& model) {
    _model = std::dynamic_pointer_cast<const models::SequenceToSequenceModel>(model);
    if (!_model)
      throw std::invalid_argument("Translator expects a model of type SequenceToSequenceModel");
    auto scoped_device_setter = _model->get_scoped_device_setter();
    _encoder = _model->make_encoder();
    _decoder = _model->make_decoder();
  }

  std::shared_ptr<const models::Model> Translator::detach_model() {
    auto model = _model;
    _encoder.reset();
    _decoder.reset();
    _model.reset();
    return model;
  }

  void Translator::assert_has_model() const {
    if (!_model)
      throw std::runtime_error("No model is attached to this translator");
  }

  void Translator::register_current_allocator() {
    if (!_allocator)
      _allocator = &ctranslate2::get_allocator(_model->device());
  }

}
