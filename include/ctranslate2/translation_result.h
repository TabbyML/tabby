#pragma once

#include <string>
#include <vector>

#include "vocabulary.h"

namespace ctranslate2 {

  template <typename T>
  class GenerationResult {
  public:
    GenerationResult(const std::vector<std::vector<T>>& hypotheses,
                     const std::vector<float>& scores,
                     const std::vector<std::vector<std::vector<float>>>* attention);

    const std::vector<T>& output() const;
    float score() const;

    size_t num_hypotheses() const;
    const std::vector<std::vector<T>>& hypotheses() const;
    const std::vector<float>& scores() const;

    const std::vector<std::vector<std::vector<float>>>& attention() const;
    bool has_attention() const;

  private:
    std::vector<std::vector<T>> _hypotheses;
    std::vector<float> _scores;
    std::vector<std::vector<std::vector<float>>> _attention;
  };

  using TranslationResult = GenerationResult<std::string>;

  TranslationResult make_translation_result(const GenerationResult<size_t>& result,
                                            const Vocabulary& vocabulary);

}
