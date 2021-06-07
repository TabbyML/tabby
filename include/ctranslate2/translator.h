#pragma once

#include <string>
#include <vector>

#include "batch_reader.h"
#include "models/sequence_to_sequence.h"
#include "generation_result.h"

namespace ctranslate2 {

  using TranslationResult = GenerationResult<std::string>;

  class Translator;
  class TranslatorPool;

  struct TranslationOptions {
    // Beam size to use for beam search (set 1 to run greedy search).
    size_t beam_size = 2;
    // Length penalty value to apply during beam search.
    float length_penalty = 0;
    // Coverage value to apply during beam search.
    float coverage_penalty = 0;
    // Biases decoding towards a given prefix, see https://arxiv.org/abs/1912.03393 --section 4.2
    // Only activates biased-decoding when beta is in range (0, 1) and SearchStrategy is set to BeamSearch.
    // The closer beta is to 1, the stronger the bias is towards the given prefix.
    //
    // If beta <= 0 and a non-empty prefix is given, then the prefix will be used as a
    // hard-prefix rather than a soft, biased-prefix.
    float prefix_bias_beta = 0;
    // Allow the beam search to exit when the first beam finishes. Otherwise, the decoding
    // continues until beam_size hypotheses are finished.
    bool allow_early_exit = true;

    // Decoding length constraints.
    size_t max_decoding_length = 250;
    size_t min_decoding_length = 1;

    // Randomly sample from the top K candidates (not compatible with beam search, set to 0
    // to sample from the full output distribution).
    size_t sampling_topk = 1;
    // High temperature increase randomness.
    float sampling_temperature = 1;

    // Allow using the vocabulary map included in the model directory, if it exists.
    bool use_vmap = false;

    // Number of hypotheses to store in the TranslationResult class (should be smaller than
    // beam_size unless return_alternatives is set).
    size_t num_hypotheses = 1;

    // Store scores in the TranslationResult class.
    bool return_scores = false;
    // Store attention vectors in the TranslationResult class.
    bool return_attention = false;

    // Return alternatives at the first unconstrained decoding position. This is typically
    // used with a target prefix to provide alternatives at a specifc location in the
    // translation.
    bool return_alternatives = false;

    // Replace unknown target tokens by the original source token with the highest attention.
    bool replace_unknowns = false;

    void validate() const;
    bool support_batch_translation() const;

  private:
    // Internal options.
    bool validated = false;
    bool rebatch_input = true;

    friend class Translator;
    friend class TranslatorPool;
  };

  // This class holds all information required to translate from a model. Copying
  // a Translator instance does not duplicate the model data and the copy can
  // be safely executed in parallel.
  class Translator {
  public:
    Translator(const std::string& model_dir,
               Device device = Device::CPU,
               int device_index = 0,
               ComputeType compute_type = ComputeType::DEFAULT);
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
    int device_index() const;
    ComputeType compute_type() const;

    const std::shared_ptr<const models::Model>& get_model() const {
      return _model;
    }

    // Change the model while keeping the same device and compute type as the previous model.
    void set_model(const std::string& model_dir);
    void set_model(models::ModelReader& model_reader);
    void set_model(const std::shared_ptr<const models::Model>& model);

    // Detach the model from this translator, which becomes unusable until set_model is called.
    void detach_model();

    // Return the memory allocator associated with this translator.
    // The allocator is registered on the first translation.
    Allocator* get_allocator() const {
      return _allocator;
    }

  private:
    void assert_has_model() const;

    std::vector<TranslationResult>
    run_batch_translation(const std::vector<std::vector<std::string>>& source,
                          const std::vector<std::vector<std::string>>& target_prefix,
                          const TranslationOptions& options);

    std::shared_ptr<const models::Model> _model;
    std::unique_ptr<layers::Encoder> _encoder;
    std::unique_ptr<layers::Decoder> _decoder;
    const models::SequenceToSequenceModel* _seq2seq_model = nullptr;
    Allocator* _allocator = nullptr;
  };

  struct Batch {
    std::vector<std::vector<std::string>> source;
    std::vector<std::vector<std::string>> target;
    std::vector<size_t> example_index;  // Index of each example in the original input.
  };

  // Rebatch the input according to the translation options.
  // This function can also reorder the examples to improve efficiency.
  std::vector<Batch>
  rebatch_input(const std::vector<std::vector<std::string>>& source,
                const std::vector<std::vector<std::string>>& target,
                size_t max_batch_size,
                BatchType batch_type = BatchType::Examples);

}
