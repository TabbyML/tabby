#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "opennmt/storage_view.h"

namespace opennmt {

  class DecoderState {
  public:
    DecoderState();
    virtual ~DecoderState() = default;
    void reset(const StorageView& memory,
               const StorageView& memory_lengths);
    void gather(const StorageView& indices);
    StorageView& get(const std::string& name);

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

    virtual StorageView& logits(size_t step, const StorageView& ids) = 0;

  protected:
    std::unique_ptr<DecoderState> _state;
  };


  void greedy_decoding(Decoder& decoder,
                       StorageView& sample_from,
                       size_t end_token,
                       size_t vocabulary_size,
                       size_t max_steps,
                       std::vector<std::vector<size_t> >& sampled_ids);
}
