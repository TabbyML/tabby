#pragma once

#include <string>
#include <vector>

#include "translation.h"
#include "models/sequence_to_sequence.h"

namespace ctranslate2 {

  // The Translator can run translations from a sequence-to-sequence model.
  // In most cases, you should prefer using the higher level TranslatorPool class which
  // supports parallel translations, asynchronous translations, and input rebatching.
  class Translator {
  public:
    Translator(const std::string& model_dir,
               Device device = Device::CPU,
               int device_index = 0,
               ComputeType compute_type = ComputeType::DEFAULT);
    Translator(models::ModelReader& model_reader,
               Device device = Device::CPU,
               int device_index = 0,
               ComputeType compute_type = ComputeType::DEFAULT);
    Translator(const std::shared_ptr<const models::Model>& model);

    // Copy constructor.
    // The copy shares the same model instance, but it can be safely used in another thread.
    Translator(const Translator& other);

    // WARNING: The translator methods are not thread-safe. To run multiple translations in
    // parallel, you should copy the Translator instance in each thread.

    TranslationResult
    translate(const std::vector<std::string>& tokens);
    TranslationResult
    translate(const std::vector<std::string>& tokens,
              const TranslationOptions& options);
    TranslationResult
    translate_with_prefix(const std::vector<std::string>& source,
                          const std::vector<std::string>& target_prefix,
                          const TranslationOptions& options);

    std::vector<TranslationResult>
    translate_batch(const std::vector<std::vector<std::string>>& tokens);
    std::vector<TranslationResult>
    translate_batch(const std::vector<std::vector<std::string>>& tokens,
                    const TranslationOptions& options);
    std::vector<TranslationResult>
    translate_batch_with_prefix(const std::vector<std::vector<std::string>>& source,
                                const std::vector<std::vector<std::string>>& target_prefix,
                                const TranslationOptions& options);

    std::vector<ScoringResult>
    score_batch(const std::vector<std::vector<std::string>>& source,
                const std::vector<std::vector<std::string>>& target,
                const ScoringOptions& options = ScoringOptions());

    Device device() const;
    int device_index() const;
    ComputeType compute_type() const;

    std::shared_ptr<const models::Model> get_model() const {
      return _replica ? _replica->model() : nullptr;
    }

    // Change the model while keeping the same device and compute type as the previous model.
    void set_model(const std::string& model_dir);
    void set_model(models::ModelReader& model_reader);
    void set_model(const std::shared_ptr<const models::Model>& model);

    // Detach the model from this translator, which becomes unusable until set_model is called.
    std::shared_ptr<const models::Model> detach_model();

  private:
    void assert_has_model() const;

    std::unique_ptr<models::SequenceToSequenceReplica> _replica;
  };

}
