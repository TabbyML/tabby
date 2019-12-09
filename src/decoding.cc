#include "ctranslate2/decoding.h"

#include <cmath>
#include <limits>
#include <map>

#include "ctranslate2/ops/ops.h"
#include "./device_dispatch.h"

namespace ctranslate2 {

  static const ops::Gather gather;

  static void split_batch_beam(StorageView& input, dim_t beam_size) {
    Shape shape = input.shape();
    shape.insert(shape.begin() + 1, beam_size);
    shape[0] /= beam_size;
    input.reshape(shape);
  }

  static void merge_batch_beam(StorageView& input) {
    Shape shape = input.shape();
    shape[0] *= shape[1];
    shape.erase(shape.begin() + 1);
    input.reshape(shape);
  }

  static void gather_batch(StorageView& data, const StorageView& indices, dim_t beam_size) {
    split_batch_beam(data, beam_size);
    gather(data, indices);
    merge_batch_beam(data);
  }

  static void tile(StorageView& input, const StorageView& repeats) {
    static const ops::Tile tile_op{};
    StorageView input_clone(std::move(input));
    tile_op(input_clone, repeats, input);
  }

  static void expand_to_beam_size(StorageView& input, dim_t beam_size) {
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

  static void expand_to_beam_size(layers::DecoderState& state, dim_t beam_size) {
    for (auto& pair : state) {
      if (!pair.second.empty())
        expand_to_beam_size(pair.second, beam_size);
    }
  }

  static void penalize_token(StorageView& log_probs, dim_t token) {
    DEVICE_DISPATCH(log_probs.device(),
                    primitives<D>::strided_fill(log_probs.data<float>() + token,
                                                static_cast<float>(-1e10),
                                                log_probs.dim(-1),
                                                log_probs.dim(0)));
  }

  void beam_search(layers::Decoder& decoder,
                   layers::DecoderState& state,
                   StorageView& sample_from,
                   StorageView& candidates,
                   const StorageView& memory,
                   const StorageView& memory_lengths,
                   dim_t start_step,
                   dim_t end_token,
                   dim_t max_length,
                   dim_t min_length,
                   dim_t beam_size,
                   size_t num_hypotheses,
                   float length_penalty,
                   std::vector<std::vector<std::vector<size_t>>>& sampled_ids,
                   std::vector<std::vector<float>>& scores,
                   std::vector<std::vector<std::vector<std::vector<float>>>>* attention) {
    PROFILE("beam_search");
    const dim_t max_step = start_step + max_length;
    Device device = memory.device();
    const dim_t batch_size = sample_from.dim(0);
    dim_t cur_batch_size = batch_size;
    const ops::TopK topk_op(beam_size);
    StorageView alive_seq(sample_from);
    alive_seq.reshape({batch_size, 1});

    expand_to_beam_size(state, beam_size);
    expand_to_beam_size(alive_seq, beam_size);

    StorageView tiled_memory(memory);
    StorageView tiled_memory_lengths(memory_lengths);
    expand_to_beam_size(tiled_memory, beam_size);
    expand_to_beam_size(tiled_memory_lengths, beam_size);

    StorageView gather_indices(DataType::DT_INT32);
    StorageView topk_ids(alive_seq);
    StorageView topk_scores;
    StorageView topk_log_probs({beam_size}, std::numeric_limits<float>::lowest());
    topk_log_probs.at<float>(0) = 0;
    tile(topk_log_probs, StorageView({1}, static_cast<int32_t>(batch_size)));

    using Result = std::pair<std::vector<size_t>, std::vector<std::vector<float>>>;
    std::vector<std::map<float, Result>> hypotheses;
    hypotheses.resize(batch_size);
    sampled_ids.clear();
    sampled_ids.resize(batch_size);
    scores.clear();
    scores.resize(batch_size);
    if (attention) {
      attention->clear();
      attention->resize(batch_size);
    }

    std::vector<bool> top_beam_finished(batch_size, false);
    std::vector<dim_t> batch_offset(batch_size);
    for (dim_t i = 0; i < batch_size; ++i) {
      batch_offset[i] = i;
      sampled_ids[i].reserve(num_hypotheses);
      scores[i].reserve(num_hypotheses);
      if (attention)
        (*attention)[i].reserve(num_hypotheses);
    }

    StorageView logits(device);
    StorageView log_probs(device);
    StorageView topk_ids_device(device, topk_ids.dtype());
    StorageView topk_scores_device(device);

    StorageView alive_attention;
    StorageView attention_step;
    StorageView attention_step_device(device);

    for (dim_t step = start_step; step < max_step; ++step) {
      // Compute log probs for the current step.
      decoder(step,
              topk_ids.to(device),
              tiled_memory,
              tiled_memory_lengths,
              state,
              &logits,
              attention ? &attention_step_device : nullptr);
      ops::LogSoftMax()(logits, log_probs);

      const dim_t vocabulary_size = log_probs.dim(-1);

      // Multiply by the current beam log probs.
      DEVICE_DISPATCH(log_probs.device(),
                      primitives<D>::add_depth_broadcast(topk_log_probs.to(device).data<float>(),
                                                         log_probs.data<float>(),
                                                         topk_log_probs.size(),
                                                         log_probs.size()));

      // Penalize by the length, if enabled.
      float length_penalty_weight = 1.0;
      if (length_penalty != 0) {
        length_penalty_weight = std::pow((5.0 + static_cast<float>(step + 1)) / 6.0, length_penalty);
        ops::Mul()(log_probs, StorageView(1.f / length_penalty_weight), log_probs);
      }

      // Penalize end_token, if configured.
      if (step < min_length)
        penalize_token(log_probs, end_token);

      // Flatten the probs into a list of candidates.
      log_probs.reshape({cur_batch_size, beam_size * vocabulary_size});

      // TopK candidates.
      topk_op(log_probs, topk_scores_device, topk_ids_device);

      topk_scores = topk_scores_device.to(Device::CPU);
      topk_ids = topk_ids_device.to(Device::CPU);
      if (attention)
        attention_step.copy_from(attention_step_device);

      topk_log_probs = topk_scores;
      // Recover the true log probs if length penalty was applied.
      if (length_penalty != 0)
        ops::Mul()(topk_log_probs, StorageView(length_penalty_weight), topk_log_probs);

      // Unflatten the ids.
      gather_indices.resize({cur_batch_size * beam_size});
      for (dim_t i = 0; i < topk_ids.size(); ++i) {
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
      topk_scores.reshape({cur_batch_size, beam_size});
      topk_ids.reshape({cur_batch_size, beam_size});
      if (attention) {
        if (alive_attention.empty())
          alive_attention = attention_step;
        else {
          gather(alive_attention, gather_indices);
          StorageView cur_alive_attention(std::move(alive_attention));
          ops::Concat(1)({&cur_alive_attention, &attention_step}, alive_attention);
        }
        alive_attention.reshape({cur_batch_size,
                                 beam_size,
                                 alive_attention.dim(1),
                                 alive_attention.dim(2)});
      }

      // Check if some hypotheses are finished.
      std::vector<bool> finished(cur_batch_size, false);
      dim_t finished_count = 0;
      for (dim_t i = 0; i < cur_batch_size; ++i) {
        const dim_t batch_id = batch_offset[i];
        for (dim_t k = 0; k < beam_size; ++k) {
          if (topk_ids.at<int32_t>({i, k}) == static_cast<int32_t>(end_token)
              || step + 1 == max_step) {
            if (k == 0)
              top_beam_finished[i] = true;
            float score = topk_scores.at<float>({i, k});
            // Prevent this beam from advancing in the next step.
            topk_log_probs.at<float>({i, k}) = -1e10;
            // Save the finished hypothesis only if it is still a candidate.
            if (hypotheses[batch_id].size() < num_hypotheses
                || -score < hypotheses[batch_id].rbegin()->first) {
              std::vector<size_t> hypothesis;
              std::vector<std::vector<float>> attn;
              const dim_t max_time = alive_seq.dim(-1);
              hypothesis.reserve(max_time);
              if (attention)
                attn.reserve(max_time);
              for (dim_t t = 1; t < max_time; ++t) {
                const int32_t id = alive_seq.at<int32_t>({i, k, t});
                if (id == static_cast<int32_t>(end_token))
                  break;
                hypothesis.push_back(id);
                if (attention) {
                  const auto* attn_vec = alive_attention.index<float>({i, k, t - 1});
                  attn.emplace_back(attn_vec, attn_vec + alive_attention.dim(-1));
                }
              }

              // Use -score as the key to iterate the map from best to worst.
              hypotheses[batch_id].emplace(std::piecewise_construct,
                                           std::forward_as_tuple(-score),
                                           std::forward_as_tuple(std::move(hypothesis),
                                                                 std::move(attn)));
            }
          }
        }

        if (top_beam_finished[i] && hypotheses[batch_id].size() >= num_hypotheses) {
          ++finished_count;
          finished[i] = true;

          // Return the "num_hypotheses" best hypotheses.
          for (auto& pair : hypotheses[batch_id]) {
            if (sampled_ids[batch_id].size() >= num_hypotheses)
              break;
            scores[batch_id].push_back(-pair.first);
            sampled_ids[batch_id].emplace_back(std::move(pair.second.first));
            if (attention) {
              (*attention)[batch_id].emplace_back(std::move(pair.second.second));
            }
          }
          hypotheses[batch_id].clear();
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
            top_beam_finished[write_index] = top_beam_finished[read_index];
            batch_offset[write_index] = batch_offset[read_index];
            ++write_index;
          }
        }
        gather(topk_ids, keep_batches);
        gather(topk_log_probs, keep_batches);
        gather(alive_seq, keep_batches);
        if (attention)
          gather(alive_attention, keep_batches);
        auto keep_batches_device = keep_batches.to(device);
        gather_batch(tiled_memory, keep_batches_device, beam_size);
        gather_batch(tiled_memory_lengths, keep_batches_device, beam_size);

        // On CPU, we reorder first and then remove finished batches. Otherwise, we remove
        // finished batches from the reorder indices and then reorder. The motivation for this
        // difference is to enable the fast in place gather on CPU for state elements that should
        // not be reordered (see Decoder::gather_state and Gather::operator()).

        if (device == Device::CPU) {
          decoder.gather_state(state, gather_indices);
          for (auto& pair : state)
            gather_batch(pair.second, keep_batches_device, beam_size);
        } else {
          gather_batch(gather_indices, keep_batches, beam_size);
          decoder.gather_state(state, gather_indices.to(device));
        }

      } else {
        decoder.gather_state(state, gather_indices.to(device));
      }

      topk_ids.reshape({cur_batch_size * beam_size, 1});
      topk_log_probs.reshape({cur_batch_size * beam_size});
      alive_seq.reshape({cur_batch_size * beam_size, alive_seq.dim(-1)});
      if (attention)
        alive_attention.reshape({cur_batch_size * beam_size,
                                 alive_attention.dim(2),
                                 alive_attention.dim(3)});
    }
  }

  void greedy_search(layers::Decoder& decoder,
                     layers::DecoderState& state,
                     StorageView& sample_from,
                     StorageView& candidates,
                     const StorageView& memory,
                     const StorageView& memory_lengths,
                     dim_t start_step,
                     dim_t end_token,
                     dim_t max_length,
                     dim_t min_length,
                     std::vector<std::vector<std::vector<size_t>>>& sampled_ids,
                     std::vector<std::vector<float>>& scores,
                     std::vector<std::vector<std::vector<std::vector<float>>>>* attention) {
    PROFILE("greedy_search");
    const dim_t max_step = start_step + max_length;
    Device device = memory.device();
    const dim_t batch_size = sample_from.dim(0);
    sample_from.reshape({batch_size, 1});

    sampled_ids.clear();
    sampled_ids.resize(batch_size);
    scores.clear();
    scores.resize(batch_size);
    if (attention) {
      attention->clear();
      attention->resize(batch_size);
    }

    StorageView alive_memory(memory);
    StorageView alive_memory_lengths(memory_lengths);

    StorageView logits(device);
    StorageView log_probs(device);
    StorageView alive({batch_size}, DataType::DT_INT32);
    std::vector<bool> finished(batch_size, false);
    std::vector<dim_t> batch_offset(batch_size);
    for (dim_t i = 0; i < batch_size; ++i) {
      batch_offset[i] = i;
      sampled_ids[i].resize(1);
      scores[i].resize(1);
      if (attention)
        (*attention)[i].resize(1);
    }

    StorageView best_ids( DataType::DT_INT32);
    StorageView best_ids_device(device, DataType::DT_INT32);
    StorageView best_probs;
    StorageView best_probs_device(device);
    StorageView attention_step;
    StorageView attention_step_device(device);

    for (dim_t step = start_step; step < max_step; ++step) {
      decoder(step,
              sample_from.to(device),
              alive_memory,
              alive_memory_lengths,
              state,
              &logits,
              attention ? &attention_step_device : nullptr);
      ops::LogSoftMax()(logits, log_probs);

      // Penalize end_token, if configured.
      if (step < min_length)
        penalize_token(log_probs, end_token);

      ops::TopK(1)(log_probs, best_probs_device, best_ids_device);
      best_probs.copy_from(best_probs_device);
      best_ids.copy_from(best_ids_device);
      if (attention)
        attention_step.copy_from(attention_step_device);

      std::vector<bool> finished_batch(log_probs.dim(0), false);
      bool one_finished = false;
      dim_t count_alive = 0;
      for (dim_t i = 0; i < log_probs.dim(0); ++i) {
        int32_t true_id = best_ids.scalar_at<int32_t>({i});
        if (!candidates.empty())
          true_id = candidates.scalar_at<int32_t>({true_id});
        dim_t batch_id = batch_offset[i];
        if (true_id == static_cast<int32_t>(end_token)) {
          finished[batch_id] = true;
          finished_batch[i] = true;
          one_finished = true;
        } else {
          sample_from.at<int32_t>(i) = true_id;
          sampled_ids[batch_id][0].push_back(true_id);
          scores[batch_id][0] += best_probs.scalar_at<float>({i});
          ++count_alive;
          if (attention) {
            const auto* attn = attention_step.index<float>({i});
            (*attention)[batch_id][0].emplace_back(attn, attn + attention_step.dim(-1));
          }
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
        auto alive_device = alive.to(device);
        decoder.gather_state(state, alive_device);
        gather(alive_memory, alive_device);
        gather(alive_memory_lengths, alive_device);
      }
    }
  }

}
