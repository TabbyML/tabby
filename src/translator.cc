#include "ctranslate2/translator.h"

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
    if (other._replica)
      set_model(other._replica->model());
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
    std::vector<std::vector<std::string>> batch_source(1, source);
    std::vector<std::vector<std::string>> batch_target_prefix;
    if (!target_prefix.empty())
      batch_target_prefix.emplace_back(target_prefix);
    return translate_batch_with_prefix(batch_source, batch_target_prefix, options)[0];
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
    return _replica->translate(source, target_prefix, options);
  }

  std::vector<ScoringResult>
  Translator::score_batch(const std::vector<std::vector<std::string>>& source,
                          const std::vector<std::vector<std::string>>& target,
                          const ScoringOptions& options) {
    assert_has_model();
    return _replica->score(source, target, options);
  }

  Device Translator::device() const {
    assert_has_model();
    return get_model()->device();
  }

  int Translator::device_index() const {
    assert_has_model();
    return get_model()->device_index();
  }

  ComputeType Translator::compute_type() const {
    assert_has_model();
    return get_model()->compute_type();
  }

  void Translator::set_model(const std::string& model_dir) {
    models::ModelFileReader model_reader(model_dir);
    set_model(model_reader);
  }

  void Translator::set_model(models::ModelReader& model_reader) {
    Device device = Device::CPU;
    int device_index = 0;
    ComputeType compute_type = ComputeType::DEFAULT;
    if (_replica) {
      device = this->device();
      device_index = this->device_index();
      compute_type = this->compute_type();
    }
    set_model(models::Model::load(model_reader, device, device_index, compute_type));
  }

  void Translator::set_model(const std::shared_ptr<const models::Model>& model) {
    _replica = model->as_sequence_to_sequence();
  }

  std::shared_ptr<const models::Model> Translator::detach_model() {
    if (!_replica)
      return nullptr;
    auto model = _replica->model();
    _replica.reset();
    return model;
  }

  void Translator::assert_has_model() const {
    if (!_replica)
      throw std::runtime_error("No model is attached to this translator");
  }

}
