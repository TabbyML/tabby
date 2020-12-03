#include "ctranslate2/generation_result.h"

namespace ctranslate2 {

  template <typename T>
  GenerationResult<T>::GenerationResult(const size_t num_hypotheses, const bool with_attention)
    : _hypotheses(num_hypotheses)
    , _scores(num_hypotheses, static_cast<float>(0))
    , _attention(with_attention ? num_hypotheses : 0) {
  }

  template <typename T>
  GenerationResult<T>::GenerationResult(std::vector<std::vector<T>> hypotheses)
    : _hypotheses(std::move(hypotheses)) {
  }

  template <typename T>
  GenerationResult<T>::GenerationResult(std::vector<std::vector<T>> hypotheses,
                                        std::vector<float> scores,
                                        std::vector<std::vector<std::vector<float>>> attention)
    : _hypotheses(std::move(hypotheses))
    , _scores(std::move(scores))
    , _attention(std::move(attention)) {
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
  void GenerationResult<T>::set_scores(std::vector<float> scores) {
    _scores = std::move(scores);
  }

  template <typename T>
  bool GenerationResult<T>::has_scores() const {
    return !_scores.empty();
  }

  template <typename T>
  const std::vector<std::vector<std::vector<float>>>& GenerationResult<T>::attention() const {
    return _attention;
  }

  template <typename T>
  void GenerationResult<T>::set_attention(std::vector<std::vector<std::vector<float>>> attention) {
    _attention = std::move(attention);
  }

  template <typename T>
  bool GenerationResult<T>::has_attention() const {
    return !_attention.empty();
  }


  template class GenerationResult<std::string>;
  template class GenerationResult<size_t>;

}
