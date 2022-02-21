#pragma once

#include <string>
#include <vector>

#include "models/sequence_to_sequence.h"

namespace ctranslate2 {

  using TranslationResult = GenerationResult<std::string>;

  struct TranslationOptions {
    // Beam size to use for beam search (set 1 to run greedy search).
    size_t beam_size = 2;
    // Length penalty value to apply during beam search (set 0 to disable).
    // If normalize_scores is enabled, the scores are normalized with:
    //   hypothesis_score /= (hypothesis_length ** length_penalty)
    // Otherwise, the length penalty is applied as described in https://arxiv.org/pdf/1609.08144.pdf.
    float length_penalty = 0;
    // Coverage value to apply during beam search (set 0 to disable).
    float coverage_penalty = 0;
    // Penalty applied to the score of previously generated tokens, as described in
    // https://arxiv.org/abs/1909.05858 (set > 1 to penalize).
    float repetition_penalty = 1;
    // Disable the generation of the unknown token.
    bool disable_unk = false;
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

    // Truncate the inputs after this many tokens (set 0 to disable truncation).
    size_t max_input_length = 1024;

    // Decoding length constraints.
    size_t max_decoding_length = 256;
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

    // Normalize the score by the hypothesis length. The hypotheses are sorted accordingly.
    bool normalize_scores = false;
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

    std::unique_ptr<const Sampler> make_sampler() const;
    std::unique_ptr<const SearchStrategy> make_search_strategy() const;
  };

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

    const std::shared_ptr<const models::Model>& get_model() const {
      return _model;
    }

    // Change the model while keeping the same device and compute type as the previous model.
    void set_model(const std::string& model_dir);
    void set_model(models::ModelReader& model_reader);
    void set_model(const std::shared_ptr<const models::Model>& model);

    // Detach the model from this translator, which becomes unusable until set_model is called.
    std::shared_ptr<const models::Model> detach_model();

    // Return the memory allocator associated with this translator.
    // The allocator is registered on the first translation.
    Allocator* get_allocator() const {
      return _allocator;
    }

  private:
    void assert_has_model() const;
    void register_current_allocator();

    std::shared_ptr<const models::Model> _model;
    std::unique_ptr<layers::Encoder> _encoder;
    std::unique_ptr<layers::Decoder> _decoder;
    const models::SequenceToSequenceModel* _seq2seq_model = nullptr;
    Allocator* _allocator = nullptr;
  };

}
