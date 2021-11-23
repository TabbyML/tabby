#pragma once

#include <numeric>
#include <string>
#include <vector>

namespace ctranslate2 {

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

}
