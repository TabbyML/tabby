#pragma once

#include <numeric>
#include <string>
#include <vector>

#include "layers/decoder.h"
#include "vocabulary.h"

namespace ctranslate2 {

  struct ScoringOptions {
    // Truncate the inputs after this many tokens (set 0 to disable truncation).
    size_t max_input_length = 1024;
    dim_t offset = 0;
  };

  struct ScoringResult {
    std::vector<std::string> tokens;
    std::vector<float> tokens_score;

    float cumulated_score() const {
      return std::accumulate(tokens_score.begin(), tokens_score.end(), 0.f);
    }

    float normalized_score() const {
      const size_t num_tokens = tokens_score.size();
      if (num_tokens == 0)
        return 0.f;
      return cumulated_score() / static_cast<float>(num_tokens);
    }
  };

  // Scores a batch of sequences.
  // The sequences are internally split into the decoder input and output sequences,
  // so they should include all tokens including the start and end tokens.
  std::vector<ScoringResult>
  score_sequences(layers::Decoder& decoder,
                  layers::DecoderState& state,
                  const std::vector<std::vector<size_t>>& sequences,
                  const Vocabulary& vocabulary,
                  const dim_t preferred_size_multiple = 1,
                  const dim_t offset=0);

}
