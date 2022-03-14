#include "ctranslate2/translation.h"

namespace ctranslate2 {

  void TranslationOptions::validate() const {
    if (num_hypotheses == 0)
      throw std::invalid_argument("num_hypotheses must be > 0");
    if (beam_size == 0)
      throw std::invalid_argument("beam_size must be > 0");
    if (num_hypotheses > beam_size && !return_alternatives)
      throw std::invalid_argument("The number of hypotheses can not be greater than the beam size");
    if (sampling_topk != 1 && beam_size != 1)
      throw std::invalid_argument("Random sampling should be used with beam_size = 1");
    if (min_decoding_length > max_decoding_length)
      throw std::invalid_argument("min_decoding_length is greater than max_decoding_length");
    if (max_decoding_length == 0)
      throw std::invalid_argument("max_decoding_length must be > 0");
    if (repetition_penalty <= 0)
      throw std::invalid_argument("repetition_penalty must be > 0");
    if (repetition_penalty != 1 && use_vmap)
      throw std::invalid_argument("repetition_penalty is currently not supported with use_vmap");
    if (prefix_bias_beta >= 1)
      throw std::invalid_argument("prefix_bias_beta must be less than 1.0");
    if (prefix_bias_beta > 0 && return_alternatives)
      throw std::invalid_argument("prefix_bias_beta is not compatible with return_alternatives");
    if (prefix_bias_beta > 0 && beam_size <= 1)
      throw std::invalid_argument("prefix_bias_beta is not compatible with greedy-search");
  }

  std::unique_ptr<const Sampler> TranslationOptions::make_sampler() const {
    if (sampling_topk == 1)
      return std::make_unique<BestSampler>();
    else
      return std::make_unique<RandomSampler>(sampling_topk, sampling_temperature);
  }

  std::unique_ptr<const SearchStrategy> TranslationOptions::make_search_strategy() const {
    if (beam_size == 1)
      return std::make_unique<GreedySearch>();
    else
      return std::make_unique<BeamSearch>(beam_size,
                                          length_penalty,
                                          coverage_penalty,
                                          prefix_bias_beta,
                                          allow_early_exit);
  }

}
