#pragma once

#include <string>
#include <vector>

#include "vocabulary.h"

namespace ctranslate2 {

  class TranslationResult {
  public:
    TranslationResult(const std::vector<std::vector<size_t>>& hypotheses,
                      const std::vector<float>& scores,
                      const Vocabulary& vocabulary);
    TranslationResult(const std::vector<std::vector<std::string>>& hypotheses,
                      const std::vector<float>& scores);

    const std::vector<std::string>& output() const;
    float score() const;

    size_t num_hypotheses() const;
    const std::vector<std::vector<std::string>>& hypotheses() const;
    const std::vector<float>& scores() const;

  private:
    std::vector<std::vector<std::string>> _hypotheses;
    std::vector<float> _scores;
  };

}
