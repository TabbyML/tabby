#pragma once

#include <string>
#include <vector>

#include "models/model.h"
#include "translation_result.h"

namespace ctranslate2 {

  struct TranslationOptions {
    size_t beam_size = 2;
    size_t num_hypotheses = 1;
    size_t max_decoding_length = 250;
    size_t min_decoding_length = 1;
    float length_penalty = 0;
    size_t sampling_topk = 1;
    float sampling_temperature = 1;
    bool use_vmap = false;
    bool return_attention = false;
  };

  // This class holds all information required to translate from a model. Copying
  // a Translator instance does not duplicate the model data and the copy can
  // be safely executed in parallel.
  class Translator {
  public:
    Translator(const std::string& model_dir, Device device, int device_index = 0);
    Translator(const std::shared_ptr<const models::Model>& model);
    Translator(const Translator& other);

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

    Device device() const;

  private:
    void make_graph();

    std::vector<TranslationResult>
    run_translation(const std::vector<std::vector<std::string>>& source,
                    const std::vector<std::vector<std::string>>& target_prefix,
                    const TranslationOptions& options);

    const std::shared_ptr<const models::Model> _model;
    std::unique_ptr<layers::Encoder> _encoder;
    std::unique_ptr<layers::Decoder> _decoder;
  };

}
