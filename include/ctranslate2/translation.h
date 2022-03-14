#pragma once

#include "decoding.h"

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

    std::unique_ptr<const Sampler> make_sampler() const;
    std::unique_ptr<const SearchStrategy> make_search_strategy() const;
  };

}
