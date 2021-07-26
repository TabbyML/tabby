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
      return cumulated_score() / static_cast<float>(tokens_score.size());
    }
  };

}
