#pragma once

#include "ctranslate2/devices.h"
#include "ctranslate2/layers/decoder.h"
#include "ctranslate2/sampling.h"
#include "ctranslate2/storage_view.h"

namespace ctranslate2 {

  struct DecodingResult {
    std::vector<std::vector<size_t>> hypotheses;
    std::vector<float> scores;
    std::vector<std::vector<std::vector<float>>> attention;
  };


  class SearchStrategy {
  public:
    virtual ~SearchStrategy() = default;
    virtual std::vector<DecodingResult>
    search(layers::Decoder& decoder,
           layers::DecoderState& state,
           const Sampler& sampler,
           const std::vector<size_t>& start_ids,
           const size_t end_id,
           const size_t unk_id,
           const dim_t start_step,
           const dim_t max_length,
           const dim_t min_length,
           const bool disable_unk = false,
           const bool normalize_scores = false,
           const bool return_scores = false,
           const bool return_attention = false,
           const size_t num_hypotheses = 1,
           const float repetition_penalty = 1,
           const dim_t no_repeat_ngram_size = 0,
           const std::vector<std::vector<size_t>>* prefix_ids = nullptr) const = 0;
  };

  class BeamSearch : public SearchStrategy {
  public:
    BeamSearch(const dim_t beam_size,
               const float length_penalty = 0,
               const float coverage_penalty = 0,
               const float prefix_bias_beta = 0,
               const bool early_exit = true);

    std::vector<DecodingResult>
    search(layers::Decoder& decoder,
           layers::DecoderState& state,
           const Sampler& sampler,
           const std::vector<size_t>& start_ids,
           const size_t end_id,
           const size_t unk_id,
           const dim_t start_step,
           const dim_t max_length,
           const dim_t min_length,
           const bool disable_unk = false,
           const bool normalize_scores = false,
           const bool return_scores = false,
           const bool return_attention = false,
           const size_t num_hypotheses = 1,
           const float repetition_penalty = 1,
           const dim_t no_repeat_ngram_size = 0,
           const std::vector<std::vector<size_t>>* prefix_ids = nullptr) const override;

  private:
    const dim_t _beam_size;
    const float _length_penalty;
    const float _coverage_penalty;
    const float _prefix_bias_beta;
    const bool _early_exit;
  };

  class BiasedDecoder {
  public:
    BiasedDecoder(const float prefix_bias_beta,
                  const std::vector<std::vector<size_t>>& prefix_ids);

    void
    decode(const dim_t cur_batch_size,
           const size_t step,
           const std::vector<dim_t>& batch_offset,
           const std::vector<std::vector<bool>>& beams_diverged_from_prefix,
           const StorageView& logits,
           StorageView& log_probs);
  private:
    StorageView _spare_beam;
    const float _prefix_bias_beta;
    std::vector<std::vector<size_t>> _prefix_ids;
  };


  class GreedySearch : public SearchStrategy {
  public:
    std::vector<DecodingResult>
    search(layers::Decoder& decoder,
           layers::DecoderState& state,
           const Sampler& sampler,
           const std::vector<size_t>& start_ids,
           const size_t end_id,
           const size_t unk_id,
           const dim_t start_step,
           const dim_t max_length,
           const dim_t min_length,
           const bool disable_unk = false,
           const bool normalize_scores = false,
           const bool return_scores = false,
           const bool return_attention = false,
           const size_t num_hypotheses = 1,
           const float repetition_penalty = 1,
           const dim_t no_repeat_ngram_size = 0,
           const std::vector<std::vector<size_t>>* prefix_ids = nullptr) const override;
  };


  struct DecodingOptions {
    size_t beam_size = 1;
    float length_penalty = 0;
    float coverage_penalty = 0;
    float repetition_penalty = 1;
    size_t no_repeat_ngram_size = 0;
    bool disable_unk = false;
    float prefix_bias_beta = 0;
    bool allow_early_exit = true;
    size_t max_length = 256;
    size_t min_length = 0;
    size_t sampling_topk = 1;
    float sampling_temperature = 1;
    size_t num_hypotheses = 1;
    bool normalize_scores = false;
    bool return_scores = false;
    bool return_attention = false;
    bool return_alternatives = false;
    float min_alternative_expansion_prob = 0;
  };

  std::vector<DecodingResult>
  decode(layers::Decoder& decoder,
         layers::DecoderState& state,
         std::vector<std::vector<size_t>> start_tokens,
         size_t end_id,
         size_t unk_id,
         const DecodingOptions& options = DecodingOptions());

}
