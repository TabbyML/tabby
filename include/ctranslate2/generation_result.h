#pragma once

#include <string>
#include <vector>

namespace ctranslate2 {

  template <typename T>
  class GenerationResult {
  public:
    GenerationResult(std::vector<std::vector<T>> hypotheses);
    GenerationResult(std::vector<std::vector<T>> hypotheses,
                     std::vector<float> scores,
                     std::vector<std::vector<std::vector<float>>> attention);

    // Construct an empty result.
    GenerationResult(const size_t num_hypotheses,
                     const bool with_attention,
                     const bool with_score);

    // Construct an uninitialized result.
    GenerationResult() = default;

    size_t num_hypotheses() const;

    const std::vector<T>& output() const;
    const std::vector<std::vector<T>>& hypotheses() const;

    float score() const;
    const std::vector<float>& scores() const;
    void set_scores(std::vector<float> scores);
    bool has_scores() const;

    const std::vector<std::vector<std::vector<float>>>& attention() const;
    void set_attention(std::vector<std::vector<std::vector<float>>> attention);
    bool has_attention() const;

  private:
    std::vector<std::vector<T>> _hypotheses;
    std::vector<float> _scores;
    std::vector<std::vector<std::vector<float>>> _attention;
  };

}
