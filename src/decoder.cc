#include "ctranslate2/decoder.h"

#include "ctranslate2/ops/ops.h"

namespace ctranslate2 {

  // Convenience functions to gather "in-place" (actually uses a temporary).
  static void gather(StorageView& input, const StorageView& indices) {
    static const ops::Gather gather_op;
    StorageView input_clone(std::move(input));
    gather_op(input_clone, indices, input);
  }
  static void gather(DecoderState& state, const StorageView& indices) {
    for (auto& pair : state.get()) {
      gather(pair.second, indices);
    }
  }


  void DecoderState::reset(const StorageView& memory,
                           const StorageView& memory_lengths) {
    reset_state("memory", memory);
    reset_state("memory_lengths", memory_lengths);
  }

  std::unordered_map<std::string, StorageView>& DecoderState::get() {
    return _states;
  }

  StorageView& DecoderState::get(const std::string& name) {
    return _states.at(name);
  }

  void DecoderState::reset_state(const std::string& name, const StorageView& state) {
    auto it = _states.find(name);
    if (it == _states.end()) {
      _states.emplace(std::piecewise_construct,
                      std::forward_as_tuple(name),
                      std::forward_as_tuple(state));
    } else {
      it->second = state;
    }
  }

  std::ostream& operator<<(std::ostream& os, const DecoderState& decoder_state) {
    for (auto& pair : decoder_state._states)
      os << pair.first << " => " << pair.second << std::endl;
    return os;
  }


  template <Device D, typename T = float>
  static void multiply_beam_probabilities(StorageView& log_probs,
                                          const StorageView& alive_log_probs) {
    size_t depth = log_probs.dim(-1);
    size_t batch_size = log_probs.size() / depth;
    for (size_t i = 0; i < batch_size; ++i) {
      primitives<D>::add(alive_log_probs.at<T>(i), log_probs.data<T>() + i * depth, depth);
    }
  }

  static void tile(StorageView& input, const StorageView& repeats) {
    static const ops::Tile tile_op;
    StorageView input_clone(std::move(input));
    tile_op(input_clone, repeats, input);
  }

  static void expand_to_beam_size(StorageView& input, size_t beam_size) {
    Shape original_shape(input.shape());
    Shape tile_shape(input.shape());
    tile_shape.insert(std::next(tile_shape.begin()), 1);
    input.reshape(tile_shape);
    StorageView repeats({input.rank()}, static_cast<int32_t>(1));
    repeats.at<int32_t>(1) = beam_size;
    tile(input, repeats);
    original_shape[0] *= beam_size;
    input.reshape(original_shape);
  }

  static void expand_to_beam_size(DecoderState& state, size_t beam_size) {
    for (auto& pair : state.get()) {
      if (!pair.second.empty())
        expand_to_beam_size(pair.second, beam_size);
    }
  }

  void beam_search(Decoder& decoder,
                   StorageView& sample_from,
                   StorageView& candidates,
                   size_t end_token,
                   size_t max_steps,
                   size_t beam_size,
                   size_t num_hypotheses,
                   float length_penalty,
                   std::vector<std::vector<std::vector<size_t>>>& sampled_ids,
                   std::vector<std::vector<float>>& scores) {
    Device device = decoder.get_state().get("memory").device();
    size_t batch_size = sample_from.dim(0);
    size_t cur_batch_size = batch_size;
    const ops::TopK topk_op(beam_size);
    StorageView alive_seq(sample_from);
    alive_seq.reshape({batch_size, 1});

    expand_to_beam_size(decoder.get_state(), beam_size);
    expand_to_beam_size(alive_seq, beam_size);

    StorageView gather_indices(DataType::DT_INT32);
    StorageView topk_ids(alive_seq);
    StorageView topk_log_probs({beam_size}, std::numeric_limits<float>::lowest());
    topk_log_probs.at<float>(0) = 0;
    tile(topk_log_probs, StorageView({1}, static_cast<int32_t>(batch_size)));

    sampled_ids.clear();
    sampled_ids.resize(batch_size);
    scores.clear();
    scores.resize(batch_size);

    std::vector<size_t> batch_offset(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      batch_offset[i] = i;
      sampled_ids[i].resize(num_hypotheses);
      scores[i].resize(num_hypotheses);
    }

    static thread_local StorageView log_probs(device);
    static thread_local StorageView logits(device);
    static thread_local StorageView topk_ids_device(device, topk_ids.dtype());
    static thread_local StorageView topk_log_probs_device(device);

    for (size_t step = 0; step < max_steps + 1; ++step) {
      // Compute log probs for the current step.
      decoder.logits(step, topk_ids, candidates, logits);
      ops::LogSoftMax()(logits, log_probs);

      size_t vocabulary_size = logits.dim(-1);

      // Multiply by the current beam log probs.
      DEVICE_DISPATCH(log_probs.device(), multiply_beam_probabilities<D>(log_probs, topk_log_probs));

      // Penalize by the length, if enabled.
      float length_penalty_weight = 1.0;
      if (length_penalty != 0) {
        length_penalty_weight = std::pow((5.0 + static_cast<float>(step + 1)) / 6.0, length_penalty);
        DEVICE_DISPATCH(log_probs.device(), primitives<D>::mul(1.f / length_penalty_weight, log_probs.data<float>(), log_probs.size()));
      }

      // Flatten the probs into a list of candidates.
      log_probs.reshape({cur_batch_size, beam_size * vocabulary_size});

      // TopK candidates.
      topk_op(log_probs, topk_log_probs_device, topk_ids_device);

      topk_log_probs = topk_log_probs_device.to(Device::CPU);
      topk_ids = topk_ids_device.to(Device::CPU);

      // Recover the true log probs if length penalty was applied.
      if (length_penalty != 0)
        primitives<>::mul(length_penalty_weight, topk_log_probs.data<float>(), topk_log_probs.size());

      // Unflatten the ids.
      topk_log_probs.reshape({cur_batch_size, beam_size});
      topk_ids.reshape({cur_batch_size, beam_size});
      gather_indices.resize({cur_batch_size, beam_size});
      for (size_t i = 0; i < topk_ids.size(); ++i) {
        auto flat_id = topk_ids.at<int32_t>(i);
        auto beam_id = flat_id / vocabulary_size;
        auto word_id = flat_id % vocabulary_size;
        auto batch_id = i / beam_size;
        if (!candidates.empty())
          word_id = candidates.at<int32_t>(word_id);
        topk_ids.at<int32_t>(i) = word_id;
        gather_indices.at<int32_t>(i) = beam_id + batch_id * beam_size;
      }

      // Check if some sentences are finished.
      std::vector<bool> finished(cur_batch_size, false);
      size_t finished_count = 0;
      for (size_t i = 0; i < cur_batch_size; ++i) {
        auto top_beam_pred_id = topk_ids.at<int32_t>({i, 0});
        if (top_beam_pred_id == static_cast<int32_t>(end_token) || step + 1 == max_steps) {
          ++finished_count;
          finished[i] = true;
          size_t batch_id = batch_offset[i];
          for (size_t k = 0; k < num_hypotheses; ++k) {
            size_t hyp_length = 0;
            for (size_t t = 1; t < alive_seq.dim(-1); ++t) {
              size_t id = alive_seq.at<int32_t>({i * beam_size + k, t});
              if (id == end_token)
                break;
              sampled_ids[batch_id][k].push_back(id);
              ++hyp_length;
            }
            scores[batch_id][k] = topk_log_probs.at<float>({i, k}) / hyp_length;
          }
        }
      }

      // If all remaining sentences are finished, no need to go further.
      if (finished_count == cur_batch_size)
        break;

      // If some sentences finished on this step, ignore them for the next step.
      if (finished_count > 0) {
        cur_batch_size -= finished_count;
        StorageView keep_batches({cur_batch_size}, DataType::DT_INT32);
        size_t write_index = 0;
        size_t read_index = 0;
        for (; read_index < finished.size(); ++read_index) {
          if (!finished[read_index]) {
            keep_batches.at<int32_t>(write_index) = read_index;
            batch_offset[write_index] = batch_offset[read_index];
            ++write_index;
          }
        }
        gather(topk_ids, keep_batches);
        gather(topk_log_probs, keep_batches);
        gather(gather_indices, keep_batches);
      }

      topk_ids.reshape({cur_batch_size * beam_size, 1});
      topk_log_probs.reshape({cur_batch_size * beam_size});
      gather_indices.reshape({cur_batch_size * beam_size});

      // Reorder hypotheses and states.
      gather(alive_seq, gather_indices);
      gather(decoder.get_state(), gather_indices);

      // Append last prediction.
      StorageView cur_alive_seq(alive_seq);
      ops::Concat(-1)({&cur_alive_seq, &topk_ids}, alive_seq);
    }
  }

  void greedy_decoding(Decoder& decoder,
                       StorageView& sample_from,
                       StorageView& candidates,
                       size_t end_token,
                       size_t max_steps,
                       std::vector<std::vector<std::vector<size_t>>>& sampled_ids,
                       std::vector<std::vector<float>>& scores) {
    Device device = decoder.get_state().get("memory").device();
    size_t batch_size = sample_from.dim(0);
    sample_from.reshape({batch_size, 1});

    sampled_ids.clear();
    sampled_ids.resize(batch_size);
    scores.clear();
    scores.resize(batch_size);

    static thread_local StorageView log_probs(device);
    static thread_local StorageView logits(device);
    StorageView alive({batch_size}, DataType::DT_INT32);
    std::vector<bool> finished(batch_size, false);
    std::vector<size_t> batch_offset(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      batch_offset[i] = i;
      sampled_ids[i].resize(1);
      scores[i].resize(1);
    }

    for (size_t step = 0; step < max_steps + 1; ++step) {
      decoder.logits(step, sample_from, candidates, logits);
      ops::LogSoftMax()(logits, log_probs);

      std::vector<bool> finished_batch(logits.dim(0), false);
      bool one_finished = false;
      size_t count_alive = 0;
      for (size_t i = 0; i < logits.dim(0); ++i) {
        size_t best = 0;
        DEVICE_DISPATCH(log_probs.device(),
                        best = primitives<D>::max_element(log_probs.index<float>({i}),
                                                          log_probs.dim(-1)));
        size_t true_id = best;
        if (!candidates.empty())
          true_id = candidates.at<int32_t>(best);
        size_t batch_id = batch_offset[i];
        if (true_id == end_token || step + 1 == max_steps) {
          finished[batch_id] = true;
          finished_batch[i] = true;
          one_finished = true;
          scores[batch_id][0] /= step;
        } else {
          sample_from.at<int32_t>(i) = true_id;
          sampled_ids[batch_id][0].push_back(true_id);
          scores[batch_id][0] += log_probs.scalar_at<float>({i, 0, best});
          ++count_alive;
        }
      }

      // No more sentences are alive, stop here.
      if (count_alive == 0)
        break;

      // Remove finished sentences from the execution.
      if (one_finished) {
        alive.resize({count_alive});
        size_t write_index = 0;
        size_t read_index = 0;
        for (; read_index < finished_batch.size(); ++read_index) {
          if (!finished_batch[read_index]) {
            batch_offset[write_index] = batch_offset[read_index];
            alive.at<int32_t>(write_index) = read_index;
            ++write_index;
          }
        }
        gather(sample_from, alive);
        gather(decoder.get_state(), alive);
      }
    }
  }

}
