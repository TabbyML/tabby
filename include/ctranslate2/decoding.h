#pragma once

#include "ctranslate2/devices.h"
#include "ctranslate2/layers/decoder.h"
#include "ctranslate2/sampling.h"
#include "ctranslate2/storage_view.h"
#include "ctranslate2/generation_result.h"

namespace ctranslate2 {

  class SearchStrategy {
  public:
    virtual ~SearchStrategy() = default;
    virtual std::vector<GenerationResult<size_t>>
    search(layers::Decoder& decoder,
           layers::DecoderState& state,
           const Sampler& sampler,
           const std::vector<size_t>& start_ids,
           const size_t end_id,
           const dim_t start_step,
           const dim_t max_length,
           const dim_t min_length,
           const std::vector<size_t>* output_ids_map,
           const bool normalize_scores = false,
           const bool return_scores = false,
           const bool return_attention = false,
           const size_t num_hypotheses = 1,
           const std::vector<std::vector<size_t>>* prefix_ids = nullptr) const = 0;
  };

  class BeamSearch : public SearchStrategy {
  public:
    BeamSearch(const dim_t beam_size,
               const float length_penalty = 0,
               const float coverage_penalty = 0,
               const float repetition_penalty = 1,
               const float prefix_bias_beta = 0,
               const bool early_exit = true);

    std::vector<GenerationResult<size_t>>
    search(layers::Decoder& decoder,
           layers::DecoderState& state,
           const Sampler& sampler,
           const std::vector<size_t>& start_ids,
           const size_t end_id,
           const dim_t start_step,
           const dim_t max_length,
           const dim_t min_length,
           const std::vector<size_t>* output_ids_map,
           const bool normalize_scores = false,
           const bool return_scores = false,
           const bool return_attention = false,
           const size_t num_hypotheses = 1,
           const std::vector<std::vector<size_t>>* prefix_ids = nullptr) const override;

  private:
    const dim_t _beam_size;
    const float _length_penalty;
    const float _coverage_penalty;
    const float _repetition_penalty;
    const float _prefix_bias_beta;
    const bool _early_exit;
  };

  class BiasedDecoder {
  public:
    BiasedDecoder() = default;

    void
    decode(const float prefix_bias_beta,
           const dim_t cur_batch_size,
           const size_t step,
           const std::vector<dim_t>& batch_offset,
           const std::vector<std::vector<bool>>& beams_diverged_from_prefix,
           const std::vector<std::vector<size_t>>& prefix_ids,
           const StorageView& logits,
           StorageView& log_probs);
  private:
    StorageView _spare_beam;
  };


  class GreedySearch : public SearchStrategy {
  public:
    std::vector<GenerationResult<size_t>>
    search(layers::Decoder& decoder,
           layers::DecoderState& state,
           const Sampler& sampler,
           const std::vector<size_t>& start_ids,
           const size_t end_id,
           const dim_t start_step,
           const dim_t max_length,
           const dim_t min_length,
           const std::vector<size_t>* output_ids_map,
           const bool normalize_scores = false,
           const bool return_scores = false,
           const bool return_attention = false,
           const size_t num_hypotheses = 1,
           const std::vector<std::vector<size_t>>* prefix_ids = nullptr) const override;
  };

  std::vector<GenerationResult<size_t>>
  decode(layers::Decoder& decoder,
         layers::DecoderState& state,
         const SearchStrategy& search_strategy,
         const Sampler& sampler,
         std::vector<size_t> start_ids,
         const std::vector<std::vector<size_t>>* prefix_ids,
         const std::vector<size_t>* output_ids_map,
         const size_t end_id,
         dim_t max_length,
         dim_t min_length,
         const size_t num_hypotheses,
         const bool return_alternatives,
         const bool return_scores,
         const bool return_attention,
         const bool normalize_scores);

}
