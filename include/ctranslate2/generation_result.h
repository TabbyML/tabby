#pragma once

#include <stdexcept>
#include <vector>

namespace ctranslate2 {

  template <typename T>
  class GenerationResult {
  public:
    std::vector<std::vector<T>> hypotheses;
    std::vector<float> scores;
    std::vector<std::vector<std::vector<float>>> attention;

    GenerationResult(std::vector<std::vector<T>> hypotheses_)
      : hypotheses(std::move(hypotheses_))
    {
    }

    GenerationResult(std::vector<std::vector<T>> hypotheses_,
                     std::vector<float> scores_,
                     std::vector<std::vector<std::vector<float>>> attention_)
      : hypotheses(std::move(hypotheses_))
      , scores(std::move(scores_))
      , attention(std::move(attention_))
    {
    }

    // Construct an empty result.
    GenerationResult(const size_t num_hypotheses,
                     const bool with_attention,
                     const bool with_score)
      : hypotheses(num_hypotheses)
      , scores(with_score ? num_hypotheses : 0, static_cast<float>(0))
      , attention(with_attention ? num_hypotheses : 0)
    {
    }

    // Construct an uninitialized result.
    GenerationResult() = default;

    const std::vector<T>& output() const {
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
