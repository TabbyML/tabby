#pragma once

#include <string>
#include <unordered_map>

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace layers {

    using DecoderState = std::unordered_map<std::string, StorageView>;

    // Base class for decoders.
    class Decoder {
    public:
      Decoder(Device device);
      virtual ~Decoder() = default;

      virtual void set_vocabulary_mask(const StorageView&) {}
      virtual void reset_vocabulary_mask() {}
      virtual DecoderState initial_state() const = 0;
      virtual void operator()(dim_t step,
                              const StorageView& ids,
                              DecoderState& state,
                              StorageView* logits = nullptr,
                              StorageView* attention = nullptr) = 0;

      // Gathers states based on indices.
      void gather_state(DecoderState& state, const StorageView& indices) const;

      Device device() const;

    protected:
      // Returns false if the state does not need to be reordered during beam search.
      virtual bool should_reorder_state(const std::string& name) const;
      // Returns the current batch size from the decoder state.
      virtual dim_t batch_size(const DecoderState& state) const;

      const Device _device;
    };

  }
}
