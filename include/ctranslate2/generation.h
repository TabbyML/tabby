#pragma once

#include <variant>
#include <vector>
#include <string>

#include "decoding.h"
#include "vocabulary.h"

namespace ctranslate2 {

  struct GenerationStepResult;

  struct GenerationOptions {
    // Beam size to use for beam search (set 1 to run greedy search).
    size_t beam_size = 1;
    // Beam search patience factor, as described in https://arxiv.org/abs/2204.05424.
    // The decoding will continue until beam_size*patience hypotheses are finished.
    float patience = 1;
    // Exponential penalty applied to the length during beam search.
    // The scores are normalized with:
    //   hypothesis_score /= (hypothesis_length ** length_penalty)
    float length_penalty = 1;
    // Penalty applied to the score of previously generated tokens, as described in
    // https://arxiv.org/abs/1909.05858 (set > 1 to penalize).
    float repetition_penalty = 1;
    // Prevent repetitions of ngrams with this size (set 0 to disable).
    size_t no_repeat_ngram_size = 0;
    // Disable the generation of the unknown token.
    bool disable_unk = false;
    // Disable the generation of some sequences of tokens.
    std::vector<std::vector<std::string>> suppress_sequences;

    // Stop the decoding on one of these tokens (defaults to the model EOS token).
    std::variant<std::string, std::vector<std::string>, std::vector<size_t>> end_token;

    // Include the end token in the result.
    bool return_end_token = false;

    // Length constraints.
    size_t max_length = 512;
    size_t min_length = 0;

    // Randomly sample from the top K candidates (set 0 to sample from the full output distribution).
    size_t sampling_topk = 1;
    // High temperature increase randomness.
    float sampling_temperature = 1;

    // Number of hypotheses to include in the result.
    size_t num_hypotheses = 1;

    // Include scores in the result.
    bool return_scores = false;

    // Return alternatives at the first unconstrained decoding position. This is typically
    // used with a prefix to provide alternatives at a specifc location.
    bool return_alternatives = false;
    // Minimum probability to expand an alternative.
    float min_alternative_expansion_prob = 0;

    // Include the input tokens in the generation result.
    bool include_prompt_in_result = true;

    // Function to call for each generated token in greedy search.
    std::function<void(GenerationStepResult)> callback = nullptr;
  };

  struct GenerationResult {
    std::vector<std::vector<std::string>> sequences;
    std::vector<std::vector<size_t>> sequences_ids;
    std::vector<float> scores;

    size_t num_sequences() const {
      return sequences.size();
    }

    bool has_scores() const {
      return !scores.empty();
    }
  };

  struct GenerationStepResult {
    size_t step;
    size_t batch_id;
    size_t token_id;
    std::string token;
    std::optional<float> log_prob;
    bool is_last;

    GenerationStepResult() = default;
    GenerationStepResult(const DecodingStepResult& result, const Vocabulary& vocabulary)
      : step(result.step)
      , batch_id(result.batch_id)
      , token_id(result.token_id)
      , token(vocabulary.to_token(result.token_id))
      , log_prob(result.log_prob)
      , is_last(result.is_last)
    {
    }
  };

  class ResolveEndToken {
  private:
    const Vocabulary& _vocabulary;

  public:
    ResolveEndToken(const Vocabulary& vocabulary)
      : _vocabulary(vocabulary)
    {
    }

    std::vector<size_t> operator()(const std::string& token) const {
      if (token.empty())
        return {_vocabulary.eos_id()};
      return {_vocabulary.to_id(token, /*allow_unk=*/false)};
    }

    std::vector<size_t> operator()(const std::vector<std::string>& tokens) const {
      std::vector<size_t> ids;
      ids.reserve(tokens.size());
      for (const auto& token : tokens)
        ids.emplace_back(_vocabulary.to_id(token, /*allow_unk=*/false));
      return ids;
    }

    std::vector<size_t> operator()(const std::vector<size_t>& tokens) const {
      return tokens;
    }
  };

}
