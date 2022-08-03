#pragma once

#include <vector>
#include <string>

namespace ctranslate2 {

  struct GenerationOptions {
    // Beam size to use for beam search (set 1 to run greedy search).
    size_t beam_size = 1;
    // Length penalty value to apply during beam search (set 0 to disable).
    // If normalize_scores is enabled, the scores are normalized with:
    //   hypothesis_score /= (hypothesis_length ** length_penalty)
    // Otherwise, the length penalty is applied as described in https://arxiv.org/pdf/1609.08144.pdf.
    float length_penalty = 0;
    // Penalty applied to the score of previously generated tokens, as described in
    // https://arxiv.org/abs/1909.05858 (set > 1 to penalize).
    float repetition_penalty = 1;
    // Prevent repetitions of ngrams with this size (set 0 to disable).
    size_t no_repeat_ngram_size = 0;
    // Disable the generation of the unknown token.
    bool disable_unk = false;
    // Allow the beam search to exit when the first beam finishes. Otherwise, the decoding
    // continues until beam_size hypotheses are finished.
    bool allow_early_exit = true;

    // Length constraints.
    size_t max_length = 512;
    size_t min_length = 0;

    // Randomly sample from the top K candidates (set 0 to sample from the full output distribution).
    size_t sampling_topk = 1;
    // High temperature increase randomness.
    float sampling_temperature = 1;

    // Number of hypotheses to include in the result (should be smaller than beam_size unless
    // return_alternatives is set).
    size_t num_hypotheses = 1;

    // Normalize the score by the hypothesis length. The hypotheses are sorted accordingly.
    bool normalize_scores = false;
    // Include scores in the result.
    bool return_scores = false;

    // Return alternatives at the first unconstrained decoding position. This is typically
    // used with a prefix to provide alternatives at a specifc location.
    bool return_alternatives = false;
    // Minimum probability to expand an alternative.
    float min_alternative_expansion_prob = 0;
  };

  struct GenerationResult {
    std::vector<std::vector<std::string>> sequences;
    std::vector<float> scores;

    GenerationResult() = default;
    GenerationResult(std::vector<std::vector<std::string>> sequences_,
                     std::vector<float> scores_)
      : sequences(sequences_)
      , scores(scores_)
    {
    }

    size_t num_sequences() const {
      return sequences.size();
    }

    bool has_scores() const {
      return !scores.empty();
    }
  };

}
