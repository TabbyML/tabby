#include "ctranslate2/layers/decoder.h"

#include <algorithm>
#include <numeric>

#include "ctranslate2/decoding_utils.h"
#include "ctranslate2/ops/ops.h"
#include "dispatch.h"

namespace ctranslate2 {
  namespace layers {

    void zero_first_timestep(StorageView& x, dim_t step) {
      if (step == 0) {
        x.zero();
      } else if (step < 0) {
        // TODO: a more direct way to set the first timestep to 0.
        const auto dtype = x.dtype();
        const auto device = x.device();
        StorageView first_step(dtype, device);
        StorageView other_steps(dtype, device);
        ops::Split(1, {1, x.dim(1) - 1})(x, first_step, other_steps);
        first_step.zero();
        ops::Concat(1)({&first_step, &other_steps}, x);
      }
    }


    Decoder::Decoder(Device device)
      : _device(device) {
    }

    void Decoder::update_state(DecoderState& state, const StorageView& alive_batches) const {
      for (auto& pair : state) {
        ops::Gather()(pair.second, alive_batches);
      }
    }

    void Decoder::update_state(DecoderState& state,
                               StorageView beam_indices,
                               const dim_t beam_size,
                               const StorageView* alive_batches) const {
      if (alive_batches) {
        split_batch_beam(beam_indices, beam_size);
        ops::Gather()(beam_indices, *alive_batches);
        merge_batch_beam(beam_indices);
      }

      for (auto& [name, value] : state) {
        if (replicate_state(name))
          ops::Gather()(value, beam_indices);
        else if (alive_batches)
          ops::Gather()(value, *alive_batches);
      }
    }

    void Decoder::replicate_state(DecoderState& state, const dim_t beam_size) const {
      for (auto& [name, value] : state) {
        if (value && replicate_state(name))
          repeat_batch(value, beam_size);
      }
    }

    dim_t Decoder::batch_size(const DecoderState& state) const {
      return state.begin()->second.dim(0);
    }

    bool Decoder::replicate_state(const std::string&) const {
      return true;
    }

    void Decoder::update_output_layer(const dim_t size_multiple,
                                      const std::vector<size_t>& restrict_ids) {
      const dim_t current_output_size = output_size();

      if (_vocabulary_size == 0)
        _vocabulary_size = current_output_size;

      std::vector<size_t> ids = restrict_ids;

      dim_t new_output_size = ids.empty() ? _vocabulary_size : ids.size();
      dim_t padding_size = 0;
      if (new_output_size % size_multiple != 0) {
        padding_size = size_multiple - (new_output_size % size_multiple);
        new_output_size += padding_size;
      }

      if (ids.empty()) {
        // Do not update the layer if the output size is unchanged.
        if (new_output_size == current_output_size)
          return;

        // Reset the output layer if the output size is the vocabulary size.
        if (new_output_size == _vocabulary_size && padding_size == 0) {
          output_layer().select_weights(nullptr);
          _to_output_word_id.clear();
          _to_original_word_id.clear();
          return;
        }

        ids.reserve(new_output_size);
        ids.resize(_vocabulary_size);
        std::iota(ids.begin(), ids.end(), size_t(0));
      }

      std::unique_ptr<StorageView> extra_bias;

      // Pad size to the next multiple.
      if (padding_size > 0) {
        ids.resize(new_output_size, size_t(0));

        // For padding positions, we bias the output to -inf for masking during softmax.
        extra_bias = std::make_unique<StorageView>(Shape{new_output_size}, output_type(), _device);
        DEVICE_AND_TYPE_DISPATCH(
          _device, output_type(),
          primitives<D>::fill(extra_bias->data<T>(),
                              T(0),
                              new_output_size - padding_size));
        DEVICE_AND_TYPE_DISPATCH(
          _device, output_type(),
          primitives<D>::fill(extra_bias->data<T>() + new_output_size - padding_size,
                              T(-1e10),
                              padding_size));
      }

      // Select weights.
      StorageView index({new_output_size}, DataType::INT32);
      for (dim_t i = 0; i < new_output_size; ++i)
        index.at<int32_t>(i) = ids[i];
      if (index.device() != _device)
        index = index.to(_device);

      output_layer().select_weights(&index, extra_bias.get());

      _to_original_word_id = std::move(ids);
      _to_output_word_id.reserve(_to_original_word_id.size());
      for (size_t i = 0; i < _to_original_word_id.size(); ++i)
        _to_output_word_id.emplace(_to_original_word_id[i], i);
    }


    void DecoderStateCache::save(std::vector<size_t> prompt, DecoderState state) {
      const std::lock_guard<std::mutex> lock(_mutex);
      _cache.emplace(std::move(prompt), std::move(state));
    }

    const DecoderState* DecoderStateCache::get(const std::vector<size_t>& prompt) const {
      const std::lock_guard<std::mutex> lock(_mutex);
      const auto it = _cache.find(prompt);
      return it == _cache.end() ? nullptr : &it->second;
    }

  }
}
