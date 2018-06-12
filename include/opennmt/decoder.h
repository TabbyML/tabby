#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "storage_view.h"

namespace opennmt {

  class DecoderState {
  public:
    DecoderState();
    virtual ~DecoderState() = default;
    virtual void reset(const StorageView& memory,
                       const StorageView& memory_lengths);
    std::unordered_map<std::string, StorageView>& get();
    StorageView& get(const std::string& name);

    friend std::ostream& operator<<(std::ostream& os, const DecoderState& decoder_state);

  protected:
    std::unordered_map<std::string, StorageView> _states;
    void add(const std::string& name, DataType dtype = DataType::DT_FLOAT);
  };

  class Decoder {
  public:
    virtual ~Decoder() = default;

    DecoderState& get_state() {
      return *_state;
    }

    virtual void logits(size_t step,
                        const StorageView& ids,
                        const StorageView& candidates,
                        StorageView& output) = 0;

  protected:
    std::unique_ptr<DecoderState> _state;
  };


  void greedy_decoding(Decoder& decoder,
                       StorageView& sample_from,
                       StorageView& candidates,
                       size_t end_token,
                       size_t max_steps,
                       std::vector<std::vector<size_t> >& sampled_ids);
  void beam_search(Decoder& decoder,
                   StorageView& sample_from,
                   StorageView& candidates,
                   size_t end_token,
                   size_t max_steps,
                   size_t beam_size,
                   float length_penalty,
                   std::vector<std::vector<size_t>>& sampled_ids);
}
