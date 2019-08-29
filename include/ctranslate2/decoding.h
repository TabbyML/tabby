#pragma once

#include "ctranslate2/layers/decoder.h"

namespace ctranslate2 {

  void greedy_search(layers::Decoder& decoder,
                     layers::DecoderState& state,
                     StorageView& sample_from,
                     StorageView& candidates,
                     const StorageView& memory,
                     const StorageView& memory_lengths,
                     size_t start_step,
                     size_t end_token,
                     size_t max_length,
                     size_t min_length,
                     std::vector<std::vector<std::vector<size_t>>>& sampled_ids,
                     std::vector<std::vector<float>>& scores,
                     std::vector<std::vector<std::vector<std::vector<float>>>>* attention = nullptr);

  void beam_search(layers::Decoder& decoder,
                   layers::DecoderState& state,
                   StorageView& sample_from,
                   StorageView& candidates,
                   const StorageView& memory,
                   const StorageView& memory_lengths,
                   size_t start_step,
                   size_t end_token,
                   size_t max_length,
                   size_t min_length,
                   size_t beam_size,
                   size_t num_hypotheses,
                   float length_penalty,
                   std::vector<std::vector<std::vector<size_t>>>& sampled_ids,
                   std::vector<std::vector<float>>& scores,
                   std::vector<std::vector<std::vector<std::vector<float>>>>* attention = nullptr);

}
