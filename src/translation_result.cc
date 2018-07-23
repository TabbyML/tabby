#include "ctranslate2/translation_result.h"

namespace ctranslate2 {

  TranslationResult::TranslationResult(const std::vector<std::vector<size_t>>& hypotheses,
                                       const std::vector<float>& scores,
                                       const Vocabulary& vocabulary)
    :_scores(scores) {
    _hypotheses.resize(hypotheses.size());
    for (size_t i = 0; i < hypotheses.size(); ++i) {
      _hypotheses[i].reserve(hypotheses[i].size());
      for (auto id : hypotheses[i])
        _hypotheses[i].push_back(vocabulary.to_token(id));
    }
  }

  TranslationResult::TranslationResult(const std::vector<std::vector<std::string>>& hypotheses,
                                       const std::vector<float>& scores)
    : _hypotheses(hypotheses)
    , _scores(scores) {
  }

  const std::vector<std::string>& TranslationResult::output() const {
    return _hypotheses[0];
  }

  float TranslationResult::score() const {
    return _scores[0];
  }

  size_t TranslationResult::num_hypotheses() const {
    return _hypotheses.size();
  }

  const std::vector<std::vector<std::string>>& TranslationResult::hypotheses() const {
    return _hypotheses;
  }

  const std::vector<float>& TranslationResult::scores() const {
    return _scores;
  }

}
