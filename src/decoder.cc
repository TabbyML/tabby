#include "ctranslate2/decoder.h"

#include "ctranslate2/ops/ops.h"

namespace ctranslate2 {

  // Convenience functions to gather "in-place" (actually uses a temporary).
  static void gather(StorageView& input, const StorageView& indices, StorageView* cache = nullptr) {
    static const ops::Gather gather_op;
    if (cache == nullptr) {
      StorageView input_clone(std::move(input));
      gather_op(input_clone, indices, input);
    } else {
      gather_op(input, indices, *cache);
      std::swap(input, *cache);
    }
  }
  static void gather(DecoderState& state, const StorageView& indices) {
    for (auto& pair : state.get()) {
      gather(pair.second, indices, state.get_cache(pair.first));
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
  StorageView* DecoderState::get_cache(const std::string& name) {
    auto it = _cache.find(name);
    if (it == _cache.end())
      return nullptr;
    return &it->second;
  }

  void DecoderState::reset_state(const std::string& name, const StorageView& state) {
    auto it = _states.find(name);
    if (it == _states.end()) {
      _states.emplace(std::piecewise_construct,
                      std::forward_as_tuple(name),
                      std::forward_as_tuple(state));
      if (state.device() != Device::CPU) {
        _cache.emplace(std::piecewise_construct,
                       std::forward_as_tuple(name),
                       std::forward_as_tuple(state.device(), state.dtype()));
      }
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

  static void tile(StorageView& input, const StorageView& repeats, StorageView* cache = nullptr) {
    static const ops::Tile tile_op;
    if (cache == nullptr) {
      StorageView input_clone(std::move(input));
      tile_op(input_clone, repeats, input);
    } else {
      tile_op(input, repeats, *cache);
      std::swap(input, *cache);
    }
  }

  static void expand_to_beam_size(StorageView& input,
                                  size_t beam_size,
                                  StorageView* cache = nullptr) {
    Shape original_shape(input.shape());
    Shape tile_shape(input.shape());
    tile_shape.insert(std::next(tile_shape.begin()), 1);
    input.reshape(tile_shape);
    StorageView repeats({input.rank()}, static_cast<int32_t>(1));
    repeats.at<int32_t>(1) = beam_size;
    tile(input, repeats, cache);
    original_shape[0] *= beam_size;
    input.reshape(original_shape);
  }

  static void expand_to_beam_size(DecoderState& state, size_t beam_size) {
    for (auto& pair : state.get()) {
      if (!pair.second.empty())
        expand_to_beam_size(pair.second, beam_size, state.get_cache(pair.first));
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

    std::vector<std::map<float, std::vector<size_t>>> hypotheses;
    hypotheses.resize(batch_size);
    sampled_ids.clear();
    sampled_ids.resize(batch_size);
    scores.clear();
    scores.resize(batch_size);

    std::vector<bool> top_beam_finished(batch_size, false);
    std::vector<size_t> batch_offset(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      batch_offset[i] = i;
      sampled_ids[i].reserve(num_hypotheses);
      scores[i].reserve(num_hypotheses);
    }

    static thread_local StorageView log_probs(device);
    static thread_local StorageView topk_ids_device(device, topk_ids.dtype());
    static thread_local StorageView topk_log_probs_device(device);
    static thread_local StorageView gather_indices_device(device, DataType::DT_INT32);

    for (size_t step = 0; step < max_steps; ++step) {
      // Compute log probs for the current step.
      decoder.log_probs(step, topk_ids.to(device), candidates, log_probs);

      size_t vocabulary_size = log_probs.dim(-1);

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
      gather_indices.resize({cur_batch_size * beam_size});
      for (size_t i = 0; i < topk_ids.size(); ++i) {
        auto flat_id = topk_ids.at<int32_t>(i);
        auto beam_id = flat_id / vocabulary_size;
        auto word_id = flat_id % vocabulary_size;
        auto batch_id = i / beam_size;
        if (!candidates.empty())
          word_id = candidates.scalar_at<int32_t>({word_id});
        topk_ids.at<int32_t>(i) = word_id;
        gather_indices.at<int32_t>(i) = beam_id + batch_id * beam_size;
      }

      // Append last prediction.
      gather(alive_seq, gather_indices);
      alive_seq.reshape({cur_batch_size, beam_size, alive_seq.dim(-1)});
      topk_ids.reshape({cur_batch_size, beam_size, 1});
      StorageView cur_alive_seq(std::move(alive_seq));
      ops::Concat(-1)({&cur_alive_seq, &topk_ids}, alive_seq);
      topk_log_probs.reshape({cur_batch_size, beam_size});
      topk_ids.reshape({cur_batch_size, beam_size});

      // Check if some hypotheses are finished.
      std::vector<bool> finished(cur_batch_size, false);
      size_t finished_count = 0;
      for (size_t i = 0; i < cur_batch_size; ++i) {
        size_t batch_id = batch_offset[i];
        for (size_t k = 0; k < beam_size; ++k) {
          if (topk_ids.at<int32_t>({i, k}) == static_cast<int32_t>(end_token)
              || step + 1 == max_steps) {
            if (k == 0)
              top_beam_finished[i] = true;
            float score = topk_log_probs.at<float>({i, k});
            // Prevent this beam from advancing in the next step.
            topk_log_probs.at<float>({i, k}) = -1e10;
            // Save the finished hypothesis only if it is still a candidate.
            if (hypotheses[batch_id].size() < num_hypotheses
                || -score < hypotheses[batch_id].rbegin()->first) {
              std::vector<size_t> hypothesis;
              hypothesis.reserve(alive_seq.dim(-1));
              for (size_t t = 1; t < alive_seq.dim(-1); ++t) {
                size_t id = alive_seq.at<int32_t>({i, k, t});
                if (id == end_token)
                  break;
                hypothesis.push_back(id);
              }

              // Use -score as the key to iterate the map from best to worst.
              hypotheses[batch_id].emplace(std::piecewise_construct,
                                           std::forward_as_tuple(-score),
                                           std::forward_as_tuple(std::move(hypothesis)));
            }
          }
        }

        if (top_beam_finished[i] && hypotheses[batch_id].size() >= num_hypotheses) {
          ++finished_count;
          finished[i] = true;

          // Return the "num_hypotheses" best hypotheses.
          for (const auto& pair : hypotheses[batch_id]) {
            if (sampled_ids[batch_id].size() >= num_hypotheses)
              break;
            sampled_ids[batch_id].emplace_back(std::move(pair.second));
            scores[batch_id].push_back(-pair.first);
          }
          hypotheses[batch_id].clear();
        }
      }

      // If all remaining sentences are finished, no need to go further.
      if (finished_count == cur_batch_size)
        break;

      // If some sentences finished on this step, ignore them for the next step.
      if (finished_count > 0) {
        gather_indices.reshape({cur_batch_size, beam_size});  // Reshape to gather on batch dim.
        cur_batch_size -= finished_count;
        StorageView keep_batches({cur_batch_size}, DataType::DT_INT32);
        size_t write_index = 0;
        size_t read_index = 0;
        for (; read_index < finished.size(); ++read_index) {
          if (!finished[read_index]) {
            keep_batches.at<int32_t>(write_index) = read_index;
            top_beam_finished[write_index] = top_beam_finished[read_index];
            batch_offset[write_index] = batch_offset[read_index];
            ++write_index;
          }
        }
        gather(topk_ids, keep_batches);
        gather(topk_log_probs, keep_batches);
        gather(alive_seq, keep_batches);
        gather(gather_indices, keep_batches);
        gather_indices.reshape({cur_batch_size * beam_size});  // Reshape back to the flat repr.
      }

      topk_ids.reshape({cur_batch_size * beam_size, 1});
      topk_log_probs.reshape({cur_batch_size * beam_size});
      alive_seq.reshape({cur_batch_size * beam_size, alive_seq.dim(-1)});

      // Reorder states.
      gather_indices_device.copy_from(gather_indices);
      gather(decoder.get_state(), gather_indices_device);
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
    StorageView alive({batch_size}, DataType::DT_INT32);
    std::vector<bool> finished(batch_size, false);
    std::vector<size_t> batch_offset(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      batch_offset[i] = i;
      sampled_ids[i].resize(1);
      scores[i].resize(1);
    }

    for (size_t step = 0; step < max_steps + 1; ++step) {
      decoder.log_probs(step, sample_from.to(device), candidates, log_probs);

      std::vector<bool> finished_batch(log_probs.dim(0), false);
      bool one_finished = false;
      size_t count_alive = 0;
      for (size_t i = 0; i < log_probs.dim(0); ++i) {
        size_t best = 0;
        DEVICE_DISPATCH(log_probs.device(),
                        best = primitives<D>::max_element(log_probs.index<float>({i}),
                                                          log_probs.dim(-1)));
        size_t true_id = best;
        if (!candidates.empty())
          true_id = candidates.scalar_at<int32_t>({best});
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
        gather(decoder.get_state(), alive.to(device));
      }
    }
  }

}
