#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "generation.h"

namespace ctranslate2 {

  struct TranslationOptions {
    // Beam size to use for beam search (set 1 to run greedy search).
    size_t beam_size = 2;
    // Beam search patience factor, as described in https://arxiv.org/abs/2204.05424.
    // The decoding will continue until beam_size*patience hypotheses are finished.
    float patience = 1;
    // Exponential penalty applied to the length during beam search.
    // The scores are normalized with:
    //   hypothesis_score /= (hypothesis_length ** length_penalty)
    float length_penalty = 1;
    // Coverage penalty weight applied during beam search.
    float coverage_penalty = 0;
    // Penalty applied to the score of previously generated tokens, as described in
    // https://arxiv.org/abs/1909.05858 (set > 1 to penalize).
    float repetition_penalty = 1;
    // Prevent repetitions of ngrams with this size (set 0 to disable).
    size_t no_repeat_ngram_size = 0;
    // Disable the generation of the unknown token.
    bool disable_unk = false;
    // Disable the generation of some sequences of tokens.
    std::vector<std::vector<std::string>> suppress_sequences;
    // Biases decoding towards a given prefix, see https://arxiv.org/abs/1912.03393 --section 4.2
    // Only activates biased-decoding when beta is in range (0, 1) and SearchStrategy is set to BeamSearch.
    // The closer beta is to 1, the stronger the bias is towards the given prefix.
    //
    // If beta <= 0 and a non-empty prefix is given, then the prefix will be used as a
    // hard-prefix rather than a soft, biased-prefix.
    float prefix_bias_beta = 0;

    // Stop the decoding on one of these tokens (defaults to the model EOS token).
    std::variant<std::string, std::vector<std::string>, std::vector<size_t>> end_token;

    // Include the end token in the result.
    bool return_end_token = false;

    // Truncate the inputs after this many tokens (set 0 to disable truncation).
    size_t max_input_length = 1024;

    // Decoding length constraints.
    size_t max_decoding_length = 256;
    size_t min_decoding_length = 1;

    // Randomly sample from the top K candidates (set 0 to sample from the full output distribution).
    size_t sampling_topk = 1;
    // High temperature increase randomness.
    float sampling_temperature = 1;

    // Allow using the vocabulary map included in the model directory, if it exists.
    bool use_vmap = false;

    // Number of hypotheses to store in the TranslationResult class.
    size_t num_hypotheses = 1;

    // Store scores in the TranslationResult class.
    bool return_scores = false;
    // Store attention vectors in the TranslationResult class.
    bool return_attention = false;

    // Return alternatives at the first unconstrained decoding position. This is typically
    // used with a target prefix to provide alternatives at a specifc location in the
    // translation.
    bool return_alternatives = false;
    // Minimum probability to expand an alternative.
    float min_alternative_expansion_prob = 0;

    // Replace unknown target tokens by the original source token with the highest attention.
    bool replace_unknowns = false;

    // Function to call for each generated token in greedy search.
    std::function<void(GenerationStepResult)> callback = nullptr;
  };

  struct TranslationResult {
    std::vector<std::vector<std::string>> hypotheses;
    std::vector<float> scores;
    std::vector<std::vector<std::vector<float>>> attention;

    TranslationResult(std::vector<std::vector<std::string>> hypotheses_)
      : hypotheses(std::move(hypotheses_))
    {
    }

    TranslationResult(std::vector<std::vector<std::string>> hypotheses_,
                      std::vector<float> scores_,
                      std::vector<std::vector<std::vector<float>>> attention_)
      : hypotheses(std::move(hypotheses_))
      , scores(std::move(scores_))
      , attention(std::move(attention_))
    {
    }

    // Construct an empty result.
    TranslationResult(const size_t num_hypotheses,
                      const bool with_attention,
                      const bool with_score)
      : hypotheses(num_hypotheses)
      , scores(with_score ? num_hypotheses : 0, static_cast<float>(0))
      , attention(with_attention ? num_hypotheses : 0)
    {
    }

    // Construct an uninitialized result.
    TranslationResult() = default;

    const std::vector<std::string>& output() const {
      if (hypotheses.empty())
        throw std::runtime_error("This result is empty");
      return hypotheses[0];
    }

    float score() const {
      if (scores.empty())
        throw std::runtime_error("This result has no scores");
      return scores[0];
    }

    size_t num_hypotheses() const {
      return hypotheses.size();
    }

    bool has_scores() const {
      return !scores.empty();
    }

    bool has_attention() const {
      return !attention.empty();
    }
  };

}
