#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace layers {

    using DecoderState = std::unordered_map<std::string, StorageView>;

    // Base class for decoders.
    class Decoder {
    public:
      Decoder(Device device);
      virtual ~Decoder() = default;

      virtual DecoderState initial_state() const = 0;
      virtual void operator()(size_t step,
                              const StorageView& ids,
                              const StorageView& candidates,
                              const StorageView& memory,
                              const StorageView& memory_lengths,
                              DecoderState& state,
                              StorageView& logits) = 0;

    protected:
      Device _device;
    };

  }


  void greedy_decoding(layers::Decoder& decoder,
                       StorageView& sample_from,
                       StorageView& candidates,
                       const StorageView& memory,
                       const StorageView& memory_lengths,
                       size_t end_token,
                       size_t max_length,
                       size_t min_length,
                       std::vector<std::vector<std::vector<size_t>>>& sampled_ids,
                       std::vector<std::vector<float>>& scores);
  void beam_search(layers::Decoder& decoder,
                   StorageView& sample_from,
                   StorageView& candidates,
                   const StorageView& memory,
                   const StorageView& memory_lengths,
                   size_t end_token,
                   size_t max_length,
                   size_t min_length,
                   size_t beam_size,
                   size_t num_hypotheses,
                   float length_penalty,
                   std::vector<std::vector<std::vector<size_t>>>& sampled_ids,
                   std::vector<std::vector<float>>& scores);
}
