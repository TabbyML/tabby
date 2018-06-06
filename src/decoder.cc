#include "opennmt/decoder.h"

#include "opennmt/ops/ops.h"

namespace opennmt {

  static void gather(StorageView& input, const StorageView& indices) {
    static const ops::Gather gather_op;
    StorageView input_clone(input);
    gather_op(input_clone, indices, input);
  }

  static void gather(DecoderState& state, const StorageView& indices) {
    for (auto& pair : state.get()) {
      gather(pair.second, indices);
    }
  }


  DecoderState::DecoderState() {
    add("memory", DataType::DT_FLOAT);
    add("memory_lengths", DataType::DT_INT32);
  }

  void DecoderState::reset(const StorageView& memory,
                           const StorageView& memory_lengths) {
    get("memory") = memory;
    get("memory_lengths") = memory_lengths;
  }

  std::unordered_map<std::string, StorageView>& DecoderState::get() {
    return _states;
  }

  StorageView& DecoderState::get(const std::string& name) {
    return _states.at(name);
  }

  void DecoderState::add(const std::string& name, DataType dtype) {
    _states.emplace(std::piecewise_construct,
                    std::forward_as_tuple(name),
                    std::forward_as_tuple(dtype));
  }

  std::ostream& operator<<(std::ostream& os, const DecoderState& decoder_state) {
    for (auto& pair : decoder_state._states)
      os << pair.first << " => " << pair.second << std::endl;
    return os;
  }


  template <typename T = float>
  static void log_probs_from_logits(const StorageView& logits, StorageView& log_probs) {
    log_probs.resize_as(logits);

    size_t depth = logits.dim(-1);
    size_t batch_size = logits.size() / depth;

    for (size_t i = 0; i < batch_size; ++i) {
      const auto* src = logits.data<T>() + i * depth;
      auto* dst = log_probs.data<T>() + i * depth;
      primitives::exp(src, dst, depth);
      T sumexp = primitives::sum(dst, depth);
      T logsumexp = std::log(sumexp);
      primitives::copy(src, dst, depth);
      primitives::sub(logsumexp, dst, depth);
    }
  }

  template <typename T = float>
  static void multiply_beam_probabilities(StorageView& log_probs,
                                          const StorageView& alive_log_probs) {
    size_t depth = log_probs.dim(-1);
    size_t batch_size = log_probs.size() / depth;
    for (size_t i = 0; i < batch_size; ++i) {
      primitives::add(alive_log_probs.at<T>(i), log_probs.data<T>() + i * depth, depth);
    }
  }

  static void tile(StorageView& input, const StorageView& repeats) {
    static const ops::Tile tile_op;
    StorageView input_clone(input);
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
                   size_t end_token,
                   size_t vocabulary_size,
                   size_t max_steps,
                   size_t beam_size,
                   float length_penalty,
                   std::vector<std::vector<size_t>>& sampled_ids) {
    size_t batch_size = sample_from.dim(0);
    size_t cur_batch_size = batch_size;
    const ops::TopK topk_op(beam_size);
    StorageView alive_seq(sample_from);

    expand_to_beam_size(decoder.get_state(), beam_size);
    expand_to_beam_size(alive_seq, beam_size);

    StorageView log_probs;
    StorageView gather_indices(DataType::DT_INT32);
    StorageView topk_ids(alive_seq);
    StorageView topk_log_probs({beam_size}, std::numeric_limits<float>::lowest());
    topk_log_probs.at<float>(0) = 0;
    tile(topk_log_probs, StorageView({1}, static_cast<int32_t>(batch_size)));

    std::vector<size_t> batch_offset(batch_size);
    for (size_t i = 0; i < batch_offset.size(); ++i)
      batch_offset[i] = i;

    sampled_ids.clear();
    sampled_ids.resize(batch_size);

    for (size_t step = 0; step < max_steps; ++step) {
      // Compute log probs for the current step.
      const auto& logits = decoder.logits(step, topk_ids);
      log_probs_from_logits(logits, log_probs);

      // Multiply by the current beam log probs.
      multiply_beam_probabilities(log_probs, topk_log_probs);

      // Penalize by the length, if enabled.
      float length_penalty_weight = 1.0;
      if (length_penalty != 0) {
        length_penalty_weight = std::pow((5.0 + static_cast<float>(step + 1)) / 6.0, length_penalty);
        primitives::mul(1.f / length_penalty_weight, log_probs.data<float>(), log_probs.size());
      }

      // Flatten the probs into a list of candidates.
      log_probs.reshape({cur_batch_size, beam_size * vocabulary_size});

      // TopK candidates.
      topk_op(log_probs, topk_log_probs, topk_ids);

      // Recover the true log probs if length penalty was applied.
      if (length_penalty != 0)
        primitives::mul(length_penalty_weight, topk_log_probs.data<float>(), topk_log_probs.size());

      // Unflatten the ids.
      topk_log_probs.reshape({cur_batch_size, beam_size});
      topk_ids.reshape({cur_batch_size, beam_size});
      gather_indices.resize({cur_batch_size, beam_size});
      for (size_t i = 0; i < topk_ids.size(); ++i) {
        const auto flat_id = topk_ids.at<int32_t>(i);
        const auto beam_id = flat_id / vocabulary_size;
        const auto word_id = flat_id % vocabulary_size;
        const auto batch_id = i / beam_size;
        topk_ids.at<int32_t>(i) = word_id;
        gather_indices.at<int32_t>(i) = beam_id + batch_id * beam_size;
      }

      // Check if some sentences are finished.
      std::vector<bool> finished(cur_batch_size, false);
      size_t finished_count = 0;
      for (size_t i = 0; i < cur_batch_size; ++i) {
        const auto pred_id = topk_ids.at<int32_t>({i, 0});
        if (pred_id == static_cast<int32_t>(end_token) || step + 1 == max_steps) {
          for (size_t t = 1; t < alive_seq.dim(-1); ++t) {
            const size_t id = alive_seq.at<int32_t>({i * beam_size, t});
            if (id == end_token)
              break;
            sampled_ids[batch_offset[i]].push_back(id);
          }
          if (step + 1 == max_steps)
            sampled_ids[batch_offset[i]].push_back(pred_id);
          ++finished_count;
          finished[i] = true;
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
                       size_t end_token,
                       size_t vocabulary_size,
                       size_t max_steps,
                       std::vector<std::vector<size_t> >& sampled_ids) {
    size_t batch_size = sample_from.dim(0);

    sampled_ids.clear();
    sampled_ids.resize(batch_size);

    StorageView probs({batch_size, vocabulary_size});
    StorageView alive({batch_size}, DataType::DT_INT32);
    std::vector<bool> finished(batch_size, false);
    std::vector<size_t> batch_offset(batch_size);
    for (size_t i = 0; i < batch_offset.size(); ++i)
      batch_offset[i] = i;
    sampled_ids.resize(batch_size);

    for (size_t step = 0; step < max_steps; ++step) {
      const auto& logits = decoder.logits(step, sample_from);
      ops::SoftMax()(logits, probs);

      std::vector<bool> finished_batch(logits.dim(0), false);
      bool one_finished = false;
      size_t count_alive = 0;
      for (size_t i = 0; i < logits.dim(0); ++i) {
        size_t best = primitives::max_element(probs.index<float>({i}), vocabulary_size);
        size_t batch_id = batch_offset[i];
        if (best == end_token) {
          finished[batch_id] = true;
          finished_batch[i] = true;
          one_finished = true;
        } else {
          sample_from.at<int32_t>(i) = best;
          sampled_ids[batch_id].push_back(best);
          ++count_alive;
        }
      }

      if (count_alive == 0)
        break;

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
