#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "storage_view.h"

namespace ctranslate2 {

  // Base class for storing decoder states as a string to StorageView map.
  class DecoderState {
  public:
    virtual ~DecoderState() = default;
    virtual void reset();
    std::unordered_map<std::string, StorageView>& get();
    StorageView& get(const std::string& name);
    StorageView* get_cache(const std::string& name);

    friend std::ostream& operator<<(std::ostream& os, const DecoderState& decoder_state);

  protected:
    std::unordered_map<std::string, StorageView> _states;
    std::unordered_map<std::string, StorageView> _cache;
    void reset_state(const std::string& name, const StorageView& state);
  };

  // Base class for decoders.
  class Decoder {
  public:
    virtual ~Decoder() = default;

    DecoderState& get_state() {
      return *_state;
    }

    virtual void log_probs(size_t step,
                           const StorageView& ids,
                           const StorageView& candidates,
                           const StorageView& memory,
                           const StorageView& memory_lengths,
                           StorageView& output) = 0;

  protected:
    std::unique_ptr<DecoderState> _state;
  };


  void greedy_decoding(Decoder& decoder,
                       StorageView& sample_from,
                       StorageView& candidates,
                       const StorageView& memory,
                       const StorageView& memory_lengths,
                       size_t end_token,
                       size_t max_steps,
                       std::vector<std::vector<std::vector<size_t>>>& sampled_ids,
                       std::vector<std::vector<float>>& scores);
  void beam_search(Decoder& decoder,
                   StorageView& sample_from,
                   StorageView& candidates,
                   const StorageView& memory,
                   const StorageView& memory_lengths,
                   size_t end_token,
                   size_t max_steps,
                   size_t beam_size,
                   size_t num_hypotheses,
                   float length_penalty,
                   std::vector<std::vector<std::vector<size_t>>>& sampled_ids,
                   std::vector<std::vector<float>>& scores);
}
