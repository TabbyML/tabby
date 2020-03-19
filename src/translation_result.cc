#include "ctranslate2/translation_result.h"

namespace ctranslate2 {

  template <typename T>
  GenerationResult<T>::GenerationResult(const std::vector<std::vector<T>>& hypotheses,
                                        const std::vector<float>& scores,
                                        const std::vector<std::vector<std::vector<float>>>* attention)
    : _hypotheses(hypotheses)
    , _scores(scores) {
    if (attention)
      _attention = *attention;
  }

  template <typename T>
  const std::vector<T>& GenerationResult<T>::output() const {
    return _hypotheses[0];
  }

  template <typename T>
  float GenerationResult<T>::score() const {
    return _scores[0];
  }

  template <typename T>
  size_t GenerationResult<T>::num_hypotheses() const {
    return _hypotheses.size();
  }

  template <typename T>
  const std::vector<std::vector<T>>& GenerationResult<T>::hypotheses() const {
    return _hypotheses;
  }

  template <typename T>
  const std::vector<float>& GenerationResult<T>::scores() const {
    return _scores;
  }

  template <typename T>
  const std::vector<std::vector<std::vector<float>>>& GenerationResult<T>::attention() const {
    return _attention;
  }

  template <typename T>
  bool GenerationResult<T>::has_attention() const {
    return !_attention.empty();
  }


  template class GenerationResult<std::string>;
  template class GenerationResult<size_t>;


  TranslationResult make_translation_result(const GenerationResult<size_t>& result,
                                            const Vocabulary& vocabulary) {
    std::vector<std::vector<std::string>> hypotheses;
    hypotheses.reserve(result.num_hypotheses());

    for (const std::vector<size_t>& ids: result.hypotheses()) {
      std::vector<std::string> tokens;
      tokens.reserve(ids.size());
      for (const size_t id : ids)
        tokens.push_back(vocabulary.to_token(id));
      hypotheses.emplace_back(std::move(tokens));
    }

    return TranslationResult(hypotheses, result.scores(), &result.attention());
  }

}
