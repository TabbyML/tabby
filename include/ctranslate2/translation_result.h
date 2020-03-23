#pragma once

#include <string>
#include <vector>

#include "vocabulary.h"

namespace ctranslate2 {

  template <typename T>
  class GenerationResult {
  public:
    GenerationResult(const size_t num_hypotheses, const bool with_attention);  // Empty result.
    GenerationResult(std::vector<std::vector<T>> hypotheses,
                     std::vector<float> scores);
    GenerationResult(std::vector<std::vector<T>> hypotheses,
                     std::vector<float> scores,
                     std::vector<std::vector<std::vector<float>>> attention);

    const std::vector<T>& output() const;
    float score() const;

    size_t num_hypotheses() const;
    const std::vector<std::vector<T>>& hypotheses() const;
    const std::vector<float>& scores() const;

    const std::vector<std::vector<std::vector<float>>>& attention() const;
    bool has_attention() const;

    friend GenerationResult<std::string>
    make_translation_result(GenerationResult<size_t>&& result,
                            const Vocabulary& vocabulary);

  private:
    std::vector<std::vector<T>> _hypotheses;
    std::vector<float> _scores;
    std::vector<std::vector<std::vector<float>>> _attention;
  };

  using TranslationResult = GenerationResult<std::string>;

}
