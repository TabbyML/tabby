#pragma once

#include <map>
#include <mutex>
#include <string>
#include <unordered_map>

#include "ctranslate2/layers/common.h"
#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace layers {

    using DecoderState = std::unordered_map<std::string, StorageView>;

    void zero_first_timestep(StorageView& x, dim_t step);

    // Base class for decoders.
    class Decoder : public Layer {
    public:
      Decoder(Device device);

      virtual DecoderState initial_state(bool iterative_decoding = true) const = 0;

      // Forwards one step.
      virtual void operator()(dim_t step,
                              const StorageView& ids,
                              DecoderState& state,
                              StorageView* logits = nullptr,
                              StorageView* attention = nullptr) = 0;

      // Forwards a full sequence.
      virtual void operator()(const StorageView& ids,
                              const StorageView& lengths,
                              DecoderState& state,
                              StorageView& logits,
                              StorageView* attention = nullptr) = 0;

      // Update the decoder state in greedy search.
      void update_state(DecoderState& state, const StorageView& alive_batches) const;

      // Update the decoder state in beam search.
      void update_state(DecoderState& state,
                        StorageView beam_indices,
                        const dim_t beam_size,
                        const StorageView* alive_batches = nullptr) const;

      // Replicate the decoder state beam_size times.
      void replicate_state(DecoderState& state, const dim_t beam_size) const;

      // Returns true if the state must be replicated beam_size times.
      virtual bool replicate_state(const std::string& name) const;

      // Restrict the output layer to a set of ids and/or resize it to a preferred size multiple.
      // Elements in restrict_ids must be unique and sorted.
      void update_output_layer(const dim_t size_multiple = 1,
                               const std::vector<size_t>& restrict_ids = {});

      bool output_layer_is_updated() const {
        return !_to_original_word_id.empty();
      }

      bool is_in_output(size_t word_id) const {
        return _to_output_word_id.find(word_id) != _to_output_word_id.end();
      }

      size_t to_output_word_id(size_t original_id) const {
        return _to_output_word_id.empty() ? original_id : _to_output_word_id.at(original_id);
      }

      size_t to_original_word_id(size_t output_id) const {
        return _to_original_word_id.empty() ? output_id : _to_original_word_id.at(output_id);
      }

      Device device() const {
        return _device;
      }

      DataType output_type() const override {
        return const_cast<Decoder&>(*this).output_layer().output_type();
      }

      dim_t output_size() const override {
        return const_cast<Decoder&>(*this).output_layer().output_size();
      }

    protected:
      // Returns the current batch size from the decoder state.
      virtual dim_t batch_size(const DecoderState& state) const;
      // Returns the output linear layer.
      virtual Dense& output_layer() = 0;

      const Device _device;

    private:
      std::vector<size_t> _to_original_word_id;
      std::unordered_map<size_t, size_t> _to_output_word_id;
      dim_t _vocabulary_size = 0;
    };


    class DecoderStateCache {
    public:
      void save(std::vector<size_t> prompt, DecoderState state);
      const DecoderState* get(const std::vector<size_t>& prompt) const;

    private:
      std::map<std::vector<size_t>, DecoderState> _cache;
      mutable std::mutex _mutex;
    };

  }
}
