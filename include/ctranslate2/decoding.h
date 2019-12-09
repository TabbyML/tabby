#pragma once

#include "ctranslate2/layers/decoder.h"

namespace ctranslate2 {

  void greedy_search(layers::Decoder& decoder,
                     layers::DecoderState& state,
                     StorageView& sample_from,
                     StorageView& candidates,
                     const StorageView& memory,
                     const StorageView& memory_lengths,
                     dim_t start_step,
                     dim_t end_token,
                     dim_t max_length,
                     dim_t min_length,
                     std::vector<std::vector<std::vector<size_t>>>& sampled_ids,
                     std::vector<std::vector<float>>& scores,
                     std::vector<std::vector<std::vector<std::vector<float>>>>* attention = nullptr);

  void beam_search(layers::Decoder& decoder,
                   layers::DecoderState& state,
                   StorageView& sample_from,
                   StorageView& candidates,
                   const StorageView& memory,
                   const StorageView& memory_lengths,
                   dim_t start_step,
                   dim_t end_token,
                   dim_t max_length,
                   dim_t min_length,
                   dim_t beam_size,
                   size_t num_hypotheses,
                   float length_penalty,
                   std::vector<std::vector<std::vector<size_t>>>& sampled_ids,
                   std::vector<std::vector<float>>& scores,
                   std::vector<std::vector<std::vector<std::vector<float>>>>* attention = nullptr);

}
