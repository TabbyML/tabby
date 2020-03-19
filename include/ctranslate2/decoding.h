#pragma once

#include "ctranslate2/layers/decoder.h"
#include "ctranslate2/sampling.h"
#include "ctranslate2/translation_result.h"

namespace ctranslate2 {

  class SearchStrategy {
  public:
    virtual ~SearchStrategy() = default;
    virtual void
    search(layers::Decoder& decoder,
           layers::DecoderState& state,
           const Sampler& sampler,
           const StorageView& start_ids,
           const dim_t start_step,
           const dim_t end_id,
           const dim_t max_length,
           const dim_t min_length,
           const std::vector<size_t>* output_ids_map,
           std::vector<std::vector<std::vector<size_t>>>& sampled_ids,
           std::vector<std::vector<float>>& scores,
           std::vector<std::vector<std::vector<std::vector<float>>>>* attention = nullptr) const = 0;
  };

  class BeamSearch : public SearchStrategy {
  public:
    BeamSearch(const dim_t beam_size,
               const float length_penalty = 0,
               const size_t num_hypotheses = 0);

    void
    search(layers::Decoder& decoder,
           layers::DecoderState& state,
           const Sampler& sampler,
           const StorageView& start_ids,
           const dim_t start_step,
           const dim_t end_id,
           const dim_t max_length,
           const dim_t min_length,
           const std::vector<size_t>* output_ids_map,
           std::vector<std::vector<std::vector<size_t>>>& sampled_ids,
           std::vector<std::vector<float>>& scores,
           std::vector<std::vector<std::vector<std::vector<float>>>>* attention = nullptr) const override;

  private:
    const dim_t _beam_size;
    const float _length_penalty;
    const size_t _num_hypotheses;
  };

  class GreedySearch : public SearchStrategy {
  public:
    void
    search(layers::Decoder& decoder,
           layers::DecoderState& state,
           const Sampler& sampler,
           const StorageView& start_ids,
           const dim_t start_step,
           const dim_t end_id,
           const dim_t max_length,
           const dim_t min_length,
           const std::vector<size_t>* output_ids_map,
           std::vector<std::vector<std::vector<size_t>>>& sampled_ids,
           std::vector<std::vector<float>>& scores,
           std::vector<std::vector<std::vector<std::vector<float>>>>* attention = nullptr) const override;
  };

  void initialize_decoder_with_prefix(const StorageView& start_ids,
                                      const std::vector<size_t>& prefix_ids,
                                      layers::Decoder& decoder,
                                      layers::DecoderState& state,
                                      std::vector<std::vector<float>>* prefix_attention);

  std::vector<GenerationResult<size_t>>
  decode(layers::Decoder& decoder,
         layers::DecoderState& state,
         const SearchStrategy& search_strategy,
         const Sampler& sampler,
         const std::vector<size_t>& start_ids,
         const std::vector<std::vector<size_t>>* prefix_ids,
         const std::vector<size_t>* output_ids_map,
         const dim_t end_id,
         const dim_t max_length,
         const dim_t min_length,
         const size_t num_hypotheses,
         const bool return_alternatives,
         const bool return_attention);

}
