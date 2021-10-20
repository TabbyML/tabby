#include "ctranslate2/decoding.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>

#include "ctranslate2/ops/ops.h"
#include "dispatch.h"

namespace ctranslate2 {

  static const ops::Gather gather;

  static void split_batch_beam(StorageView& input, dim_t beam_size) {
    Shape shape = input.shape();
    shape.insert(shape.begin() + 1, beam_size);
    shape[0] /= beam_size;
    input.reshape(std::move(shape));
  }

  static void merge_batch_beam(StorageView& input) {
    Shape shape = input.shape();
    shape[0] *= shape[1];
    shape.erase(shape.begin() + 1);
    input.reshape(std::move(shape));
  }

  static void gather_batch(StorageView& data, const StorageView& indices, dim_t beam_size) {
    split_batch_beam(data, beam_size);
    gather(data, indices);
    merge_batch_beam(data);
  }

  static void expand_to_beam_size(StorageView& input, dim_t beam_size) {
    input.expand_dims(1);
    ops::Tile(/*axis=*/1, beam_size)(input);
    merge_batch_beam(input);
  }

  static void expand_to_beam_size(layers::DecoderState& state, dim_t beam_size) {
    for (auto& pair : state) {
      if (!pair.second.empty())
        expand_to_beam_size(pair.second, beam_size);
    }
  }

  static void penalize_token(StorageView& log_probs, const size_t id) {
    DEVICE_AND_TYPE_DISPATCH(log_probs.device(), log_probs.dtype(),
                             primitives<D>::strided_fill(log_probs.data<T>() + id,
                                                         static_cast<T>(-1e10),
                                                         log_probs.dim(-1),
                                                         log_probs.dim(0)));
  }

  static void update_sample_with_prefix(const dim_t step,
                                        StorageView& sampled_ids,
                                        StorageView& sampled_scores,
                                        const std::vector<std::vector<size_t>>& prefix_ids,
                                        const size_t end_id,
                                        const std::vector<dim_t>& batch_offset) {
    const dim_t batch_size = sampled_scores.dim(0);
    const dim_t beam_size = sampled_scores.dim(1);
    for (dim_t i = 0; i < batch_size; ++i) {
      const dim_t batch_id = batch_offset[i];
      const auto& prefix = prefix_ids[batch_id];
      const dim_t prefix_length = prefix.size();
      if (step < prefix_length) {
        for (dim_t k = 0; k < beam_size; ++k) {
          sampled_ids.at<int32_t>({i, k}) = prefix[step];
          // Set the highest log score for the first beam and penalize the others.
          TYPE_DISPATCH(sampled_scores.dtype(),
                        sampled_scores.at<T>({i, k}) = (k == 0 ? 0 : T(-1e10)));
        }
      } else if (step == prefix_length) {
        // At the first unconstrained decoding step, only the first beam is expanded.
        // It happens that </s> appears in the topk, especially when k is large. This
        // can produce incorrect and short predictions that dominate others (see issue
        // #277). To mitigate this issue, we penalize </s> in secondary beams.
        for (dim_t k = 1; k < beam_size; ++k) {
          auto& sampled_id = sampled_ids.at<int32_t>({i, k});
          if (static_cast<size_t>(sampled_id) == end_id) {
            // Assign a token different than </s> and with a low probability.
            sampled_id = 0;
            TYPE_DISPATCH(sampled_scores.dtype(),
                          sampled_scores.at<T>({i, k}) = T(-1e10));
          }
        }
      }
    }
  }

  template <typename T>
  static void initialize_cum_log_probs(StorageView& cum_log_probs,
                                       const dim_t batch_size,
                                       const dim_t beam_size) {
    const dim_t size = batch_size * beam_size;
    cum_log_probs.resize({size});
    auto* data = cum_log_probs.data<T>();
    for (dim_t i = 0; i < size; ++i) {
      data[i] = (i % beam_size == 0 ? T(0) : std::numeric_limits<T>::lowest());
    }
  }

  // Sort hypotheses from best to worst score, in the limit of max_hypotheses.
  template <typename T>
  static inline void sort_hypotheses(GenerationResult<T>& result,
                                     size_t max_hypotheses,
                                     bool keep_scores) {
    std::vector<size_t> idx(result.num_hypotheses());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&result](size_t i1, size_t i2) { return result.scores[i1] > result.scores[i2]; });

    if (max_hypotheses < idx.size())
      idx.resize(max_hypotheses);

    result.hypotheses = index_vector(result.hypotheses, idx);
    if (keep_scores)
      result.scores = index_vector(result.scores, idx);
    else
      result.scores.clear();
    if (result.has_attention())
      result.attention = index_vector(result.attention, idx);
  }


  void BiasedDecoder::decode(const float prefix_bias_beta,
                             const dim_t cur_batch_size,
                             const size_t step,
                             const std::vector<dim_t>& batch_offset,
                             const std::vector<std::vector<bool>>& beams_diverged_from_prefix,
                             const std::vector<std::vector<size_t>>& prefix_ids,
                             const StorageView& logits,
                             StorageView& log_probs) {
    const dim_t num_beams = logits.dim(0);
    const Device device = logits.device();
    const DataType dtype = logits.dtype();

    if (_spare_beam.dtype() != dtype || _spare_beam.device() != device) {
      _spare_beam = StorageView(device, dtype);
    }

    std::vector<StorageView> logit_beam_view_storage(num_beams, StorageView(device, dtype));
    std::vector<StorageView*> logit_beam_views(num_beams);
    std::vector<StorageView> log_prob_beam_view_storage(num_beams, StorageView(device, dtype));
    std::vector<StorageView*> log_prob_beam_views(num_beams);
    for (dim_t i = 0; i < num_beams; ++i) {
      logit_beam_views[i] = &(logit_beam_view_storage[i]);
      log_prob_beam_views[i] = &(log_prob_beam_view_storage[i]);
    }
    ops::Split(0, /*no_copy=*/true)(logits, logit_beam_views);
    log_probs.resize_as(logits);
    log_probs.reshape(logits.shape());
    ops::Split(0, /*no_copy=*/true)(log_probs, log_prob_beam_views);

    // Scalar's need to be allocated on CPUs.
    StorageView scalar_discount(1 - prefix_bias_beta, Device::CPU);
    assert (num_beams % cur_batch_size == 0);
    const dim_t cur_beam_size = num_beams / cur_batch_size;
    for (dim_t b = 0; b < num_beams; ++b) {
      StorageView &logit_beam = *(logit_beam_views[b]);
      StorageView &log_prob_beam = *(log_prob_beam_views[b]);
      const dim_t index_batch = b / cur_beam_size;
      const dim_t index_beam = b % cur_beam_size;
      const auto& prefix = prefix_ids[batch_offset[index_batch]];
      if (static_cast<size_t>(step) < prefix.size()
          && !(beams_diverged_from_prefix[index_batch][index_beam])) {
        ops::SoftMax()(logit_beam, log_prob_beam);
        ops::Mul()(log_prob_beam,
                   scalar_discount.to(log_prob_beam.dtype()),
                   _spare_beam);
        const size_t biased_word_id = prefix[step];
        StorageView spare_scalar_view;
        TYPE_DISPATCH(
          _spare_beam.dtype(),
          spare_scalar_view = StorageView({1}, _spare_beam.data<T>() + biased_word_id, device));
        const StorageView spare_scalar_copy(spare_scalar_view);
        StorageView beta_scalar;
        TYPE_DISPATCH(
          _spare_beam.dtype(),
          // Scalar's need to be allocated on CPUs.
          beta_scalar = StorageView(static_cast<T>(prefix_bias_beta), Device::CPU));
        ops::Add()(spare_scalar_copy, beta_scalar, spare_scalar_view);
        ops::Log()(_spare_beam, log_prob_beam);
      } else {
        ops::LogSoftMax()(logit_beam, log_prob_beam);
      }
    }
  }


  BeamSearch::BeamSearch(const dim_t beam_size,
                         const float length_penalty,
                         const float coverage_penalty,
                         const float prefix_bias_beta,
                         const bool early_exit)
    : _beam_size(beam_size)
    , _length_penalty(length_penalty)
    , _coverage_penalty(coverage_penalty)
    , _prefix_bias_beta(prefix_bias_beta)
    , _early_exit(early_exit)
  {
  }

  std::vector<GenerationResult<size_t>>
  BeamSearch::search(layers::Decoder& decoder,
                     layers::DecoderState& state,
                     const Sampler& sampler,
                     const std::vector<size_t>& start_ids,
                     const size_t end_id,
                     const dim_t start_step,
                     const dim_t max_length,
                     const dim_t min_length,
                     const std::vector<size_t>* output_ids_map,
                     const bool normalize_scores,
                     const bool return_scores,
                     const bool return_attention,
                     const size_t num_hypotheses,
                     const std::vector<std::vector<size_t>>* prefix_ids) const {
    PROFILE("beam_search");
    const dim_t min_step = start_step + min_length;
    const dim_t max_step = start_step + max_length;
    const Device device = decoder.device();
    const DataType dtype = decoder.output_type();
    const bool expand_after_first_step = (device == Device::CPU);
    const dim_t batch_size = start_ids.size();
    dim_t cur_batch_size = batch_size;

    StorageView gather_indices(DataType::INT32);
    StorageView topk_ids({batch_size}, DataType::INT32);
    StorageView topk_scores(dtype);
    StorageView topk_log_probs(dtype);

    std::vector<bool> top_beam_finished(batch_size, false);
    std::vector<dim_t> batch_offset(batch_size);
    std::vector<GenerationResult<size_t>> results(batch_size);
    for (dim_t i = 0; i < batch_size; ++i) {
      batch_offset[i] = i;
      topk_ids.at<int32_t>(i) = start_ids[i];
    }

    if (!expand_after_first_step) {
      expand_to_beam_size(state, _beam_size);
      expand_to_beam_size(topk_ids, _beam_size);
      TYPE_DISPATCH(dtype, initialize_cum_log_probs<T>(topk_log_probs, batch_size, _beam_size));
    }

    std::vector<std::vector<bool>> beams_diverged_from_prefix;
    bool bias_towards_prefix = prefix_ids && _prefix_bias_beta > 0;
    if (bias_towards_prefix) {
      beams_diverged_from_prefix = std::vector<std::vector<bool>>(
        batch_size, std::vector<bool>(_beam_size, false));
    }
    const bool use_hard_prefix = prefix_ids && !bias_towards_prefix;

    StorageView logits(dtype, device);
    StorageView alive_seq(topk_ids.dtype());
    StorageView alive_attention;
    StorageView attention_step;
    StorageView attention_step_device(dtype, device);

    StorageView coverage;
    std::unique_ptr<BiasedDecoder> biased_decoder;
    for (dim_t step = start_step; step < max_step; ++step) {
      // Compute log probs for the current step.
      decoder(step,
              topk_ids.to(device),
              state,
              &logits,  // output shape: (cur_batch_size*beam_size x vocab_size), if not expanded beam_size is 1
              (return_attention || _coverage_penalty != 0) ? &attention_step_device : nullptr);

      StorageView log_probs(dtype, device);
      if (bias_towards_prefix) {
        if (!biased_decoder) {
          biased_decoder = std::make_unique<BiasedDecoder>();
        }
        biased_decoder->decode(_prefix_bias_beta,
                               cur_batch_size,
                               step,
                               batch_offset,
                               beams_diverged_from_prefix,
                               *prefix_ids,
                               logits,
                               log_probs);
      } else {
        ops::LogSoftMax()(logits);
        log_probs.shallow_copy(logits);
      }

      const dim_t vocabulary_size = log_probs.dim(-1);
      const bool is_expanded = (!expand_after_first_step || step > start_step);

      // Multiply by the current beam log probs.
      if (is_expanded) {
        DEVICE_AND_TYPE_DISPATCH(log_probs.device(), log_probs.dtype(),
                                 primitives<D>::add_depth_broadcast(topk_log_probs.to(device).data<T>(),
                                                                    log_probs.data<T>(),
                                                                    topk_log_probs.size(),
                                                                    log_probs.size()));
      }

      // Penalize end_id, if configured.
      if (step < min_step)
        penalize_token(log_probs, end_id);

      // Flatten the probs into a list of candidates.
      log_probs.reshape({cur_batch_size, -1});

      // TopK candidates.
      sampler(log_probs, topk_ids, topk_scores, _beam_size);
      if (use_hard_prefix)
        update_sample_with_prefix(step, topk_ids, topk_scores, *prefix_ids, end_id, batch_offset);

      topk_log_probs = topk_scores;

      // Unflatten the ids.
      gather_indices.resize({cur_batch_size * _beam_size});
      std::vector<std::vector<bool>> prev_beams_diverged_from_prefix;
      if (bias_towards_prefix)
        prev_beams_diverged_from_prefix = beams_diverged_from_prefix;
      for (dim_t i = 0; i < topk_ids.size(); ++i) {
        auto flat_id = topk_ids.at<int32_t>(i);
        auto beam_id = flat_id / vocabulary_size;
        auto word_id = flat_id % vocabulary_size;
        auto batch_id = i / _beam_size;
        if (output_ids_map)
          word_id = output_ids_map->at(word_id);
        if (bias_towards_prefix) {
          const auto& prefix = (*prefix_ids)[batch_offset[batch_id]];
          bool diverged = true;
          if (static_cast<size_t>(step) < prefix.size()) {
            diverged = (prev_beams_diverged_from_prefix[batch_id][beam_id]
                        || static_cast<size_t>(word_id) != prefix[step]);
          }
          beams_diverged_from_prefix[batch_id][i % _beam_size] = diverged;
        }
        topk_ids.at<int32_t>(i) = word_id;
        // On the first step, batches are not yet replicated beam_size times.
        gather_indices.at<int32_t>(i) = (is_expanded
                                         ? beam_id + batch_id * _beam_size
                                         : batch_id);
      }

      // Append last prediction.
      topk_ids.reshape({cur_batch_size, _beam_size, 1});
      if (alive_seq) {
        gather(alive_seq, gather_indices);
        alive_seq.reshape({cur_batch_size, _beam_size, alive_seq.dim(-1)});
        StorageView cur_alive_seq(std::move(alive_seq));
        ops::Concat(-1)({&cur_alive_seq, &topk_ids}, alive_seq);
      } else {
        alive_seq = topk_ids;
      }

      topk_log_probs.reshape({cur_batch_size, _beam_size});
      topk_scores.reshape({cur_batch_size, _beam_size});
      topk_ids.reshape({cur_batch_size, _beam_size});

      if (attention_step_device) {
        attention_step.copy_from(attention_step_device.to_float());
        if (!is_expanded) {
          expand_to_beam_size(attention_step, _beam_size);
        }
      }

      if (_coverage_penalty != 0) {
        if (!coverage) {
          coverage = attention_step;
        } else {
          gather(coverage, gather_indices);
          ops::Add()(attention_step, coverage, coverage);
        }
        StorageView tmp;
        ops::Min()(coverage, 1.0f, tmp);
        ops::Log()(tmp, tmp);
        tmp.reshape({-1, tmp.dim(-1)});
        StorageView penalty;
        ops::MatMul()(tmp, StorageView({tmp.dim(-1), 1}, 1.0f), penalty);
        ops::Mul()(penalty, StorageView(_coverage_penalty), penalty);
        ops::Add()(penalty.to(topk_scores.dtype()), topk_scores, topk_scores);
      }

      if (return_attention) {
        if (!alive_attention) {
          alive_attention = attention_step;
        } else {
          gather(alive_attention, gather_indices);
          StorageView cur_alive_attention(std::move(alive_attention));
          ops::Concat(1)({&cur_alive_attention, &attention_step}, alive_attention);
        }
        alive_attention.reshape({cur_batch_size,
                                 _beam_size,
                                 alive_attention.dim(1),
                                 alive_attention.dim(2)});
      }

      // Check if some hypotheses are finished.
      std::vector<int32_t> non_finished_index;
      non_finished_index.reserve(cur_batch_size);

      const dim_t max_time = alive_seq.dim(-1);

      for (dim_t i = 0; i < cur_batch_size; ++i) {
        const dim_t batch_id = batch_offset[i];
        auto& result = results[batch_id];
        for (dim_t k = 0; k < _beam_size; ++k) {
          if (topk_ids.at<int32_t>({i, k}) == static_cast<int32_t>(end_id)
              || step + 1 == max_step) {
            if (k == 0)
              top_beam_finished[i] = true;

            float score = topk_scores.scalar_at<float>({i, k});
            if (_length_penalty != 0) {
              const float base = normalize_scores ? max_time : (5.f + max_time) / 6.f;
              score /= std::pow(base, _length_penalty);
            } else if (normalize_scores) {
              score /= max_time;
            }

            // Prevent this beam from advancing in the next step.
            TYPE_DISPATCH(dtype, topk_log_probs.at<T>({i, k}) = T(-1e10));

            {
              std::vector<size_t> hypothesis;
              std::vector<std::vector<float>> attn;
              hypothesis.reserve(max_time);
              if (return_attention)
                attn.reserve(max_time);
              for (dim_t t = 0; t < max_time; ++t) {
                const int32_t id = alive_seq.at<int32_t>({i, k, t});
                hypothesis.push_back(id);
                if (return_attention) {
                  const auto* attn_vec = alive_attention.index<float>({i, k, t});
                  attn.emplace_back(attn_vec, attn_vec + alive_attention.dim(-1));
                }
                if (id == static_cast<int32_t>(end_id))
                  break;
              }

              result.scores.emplace_back(score);
              result.hypotheses.emplace_back(std::move(hypothesis));
              if (return_attention)
                result.attention.emplace_back(std::move(attn));
            }
          }
        }

        const bool is_finished = (
          _early_exit
          ? top_beam_finished[i] && result.num_hypotheses() >= num_hypotheses
          : result.num_hypotheses() >= static_cast<size_t>(_beam_size));

        if (is_finished) {
          sort_hypotheses(result, num_hypotheses, return_scores);
        } else {
          non_finished_index.emplace_back(i);
        }
      }

      const dim_t next_batch_size = non_finished_index.size();

      // If all remaining sentences are finished, no need to go further.
      if (next_batch_size == 0) {
        if (!is_expanded) {
          // We should ensure that states are replicated before exiting this function.
          expand_to_beam_size(state, _beam_size);
        }
        break;
      }

      // If some sentences finished on this step, ignore them for the next step.
      if (next_batch_size != cur_batch_size) {
        cur_batch_size = next_batch_size;
        batch_offset = index_vector(batch_offset, non_finished_index);
        top_beam_finished = index_vector(top_beam_finished, non_finished_index);
        if (bias_towards_prefix)
          beams_diverged_from_prefix = index_vector(beams_diverged_from_prefix, non_finished_index);

        StorageView keep_batches({cur_batch_size}, non_finished_index);
        gather(topk_ids, keep_batches);
        gather(topk_log_probs, keep_batches);
        gather(alive_seq, keep_batches);
        if (return_attention)
          gather(alive_attention, keep_batches);
        if (_coverage_penalty != 0)
          gather_batch(coverage, keep_batches, _beam_size);

        // On CPU, we reorder first and then remove finished batches. Otherwise, we remove
        // finished batches from the reorder indices and then reorder. The motivation for this
        // difference is to enable the fast in place gather on CPU for state elements that should
        // not be reordered (see Decoder::gather_state and Gather::operator()).

        if (device == Device::CPU) {
          decoder.gather_state(state, gather_indices);
          for (auto& pair : state)
            gather_batch(pair.second, keep_batches, _beam_size);
        } else {
          gather_batch(gather_indices, keep_batches, _beam_size);
          decoder.gather_state(state, gather_indices.to(device));
        }

      } else {
        decoder.gather_state(state, gather_indices.to(device));
      }

      topk_ids.reshape({cur_batch_size * _beam_size});
      topk_log_probs.reshape({cur_batch_size * _beam_size});
      alive_seq.reshape({cur_batch_size * _beam_size, alive_seq.dim(-1)});
      if (return_attention)
        alive_attention.reshape({cur_batch_size * _beam_size,
                                 alive_attention.dim(2),
                                 alive_attention.dim(3)});
      if (bias_towards_prefix) {
        bias_towards_prefix = false;
        for (const auto& batch : beams_diverged_from_prefix) {
          for (const bool beam_diverged : batch) {
            if (!beam_diverged) {
              bias_towards_prefix = true;
              break;
            }
          }
          if (bias_towards_prefix)
            break;
        }
      }
    }

    return results;
  }

  std::vector<GenerationResult<size_t>>
  GreedySearch::search(layers::Decoder& decoder,
                       layers::DecoderState& state,
                       const Sampler& sampler,
                       const std::vector<size_t>& start_ids,
                       const size_t end_id,
                       const dim_t start_step,
                       const dim_t max_length,
                       const dim_t min_length,
                       const std::vector<size_t>* output_ids_map,
                       const bool normalize_scores,
                       const bool return_scores,
                       const bool return_attention,
                       const size_t,
                       const std::vector<std::vector<size_t>>* prefix_ids) const {
    PROFILE("greedy_search");
    const dim_t min_step = start_step + min_length;
    const dim_t max_step = start_step + max_length;
    const Device device = decoder.device();
    const DataType dtype = decoder.output_type();
    const dim_t batch_size = start_ids.size();
    StorageView sample_from({batch_size}, DataType::INT32);

    StorageView logits(dtype, device);
    std::vector<dim_t> batch_offset(batch_size);
    std::vector<GenerationResult<size_t>> results;
    results.reserve(batch_size);
    for (dim_t i = 0; i < batch_size; ++i) {
      batch_offset[i] = i;
      sample_from.at<int32_t>(i) = start_ids[i];
      results.emplace_back(/*num_hypotheses=*/1, return_attention, return_scores);
    }

    StorageView best_ids(DataType::INT32);
    StorageView best_probs(dtype);
    StorageView attention_step;
    StorageView attention_step_device(dtype, device);

    for (dim_t step = start_step; step < max_step; ++step) {
      decoder(step,
              sample_from.to(device),
              state,
              &logits,
              return_attention ? &attention_step_device : nullptr);

      // Compute log probs only if scores should be returned.
      StorageView log_probs(dtype, device);
      if (return_scores)
        ops::LogSoftMax()(logits);
      log_probs.shallow_copy(logits);

      // Penalize end_id, if configured.
      if (step < min_step)
        penalize_token(log_probs, end_id);

      sampler(log_probs, best_ids, best_probs);
      if (prefix_ids)
        update_sample_with_prefix(step, best_ids, best_probs, *prefix_ids, end_id, batch_offset);
      if (return_attention)
        attention_step.copy_from(attention_step_device.to_float());

      const dim_t cur_batch_size = log_probs.dim(0);
      std::vector<int32_t> non_finished_index;
      non_finished_index.reserve(cur_batch_size);

      for (dim_t i = 0; i < cur_batch_size; ++i) {
        int32_t true_id = best_ids.scalar_at<int32_t>({i});
        if (output_ids_map)
          true_id = output_ids_map->at(true_id);
        dim_t batch_id = batch_offset[i];
        results[batch_id].hypotheses[0].push_back(true_id);
        if (return_scores) {
          results[batch_id].scores[0] += best_probs.scalar_at<float>({i});
        }
        if (return_attention) {
          const auto* attn = attention_step.index<float>({i});
          results[batch_id].attention[0].emplace_back(attn, attn + attention_step.dim(-1));
        }
        if (true_id != static_cast<int32_t>(end_id)) {
          non_finished_index.emplace_back(i);
          sample_from.at<int32_t>(i) = true_id;
        }
      }

      const dim_t count_alive = non_finished_index.size();

      // No more sentences are alive, stop here.
      if (count_alive == 0)
        break;

      // Remove finished sentences from the execution.
      if (count_alive != cur_batch_size) {
        batch_offset = index_vector(batch_offset, non_finished_index);

        StorageView alive({count_alive}, non_finished_index);
        gather(sample_from, alive);
        auto alive_device = alive.to(device);
        decoder.gather_state(state, alive_device);
      }
    }

    if (return_scores && normalize_scores) {
      for (auto& result : results)
        result.scores[0] /= result.hypotheses[0].size();
    }

    return results;
  }

  static void initialize_decoder_with_prefix(layers::Decoder& decoder,
                                             layers::DecoderState& state,
                                             size_t start_id,
                                             const std::vector<size_t>& prefix_ids,
                                             std::vector<std::vector<float>>* prefix_attention) {
    const Device device = decoder.device();
    const size_t prefix_size = prefix_ids.size();

    StorageView input({1}, DataType::INT32);
    StorageView attention(device);
    if (prefix_attention)
      prefix_attention->reserve(prefix_size);

    input.at<int32_t>(0) = start_id;
    for (size_t i = 0; i < prefix_size; ++i) {
      decoder(i,
              input.to(device),
              state,
              /*logits=*/nullptr,
              prefix_attention ? &attention : nullptr);
      if (prefix_attention)
        prefix_attention->emplace_back(attention.to_vector<float>());
      input.at<int32_t>(0) = prefix_ids[i];
    }
  }

  std::vector<GenerationResult<size_t>>
  decode(layers::Decoder& decoder,
         layers::DecoderState& state,
         const SearchStrategy& search_strategy,
         const Sampler& sampler,
         std::vector<size_t> start_ids,
         const std::vector<std::vector<size_t>>* prefix_ids,
         const std::vector<size_t>* output_ids_map,
         const size_t end_id,
         dim_t max_length,
         dim_t min_length,
         const size_t num_hypotheses,
         const bool return_alternatives,
         const bool return_scores,
         const bool return_attention,
         const bool normalize_scores) {
    const size_t batch_size = start_ids.size();
    dim_t start_step = 0;

    std::vector<GenerationResult<size_t>> expansion_results;
    if (return_alternatives) {
      std::vector<std::vector<std::vector<float>>> prefix_attention;
      if (prefix_ids) {
        if (prefix_ids->size() > 1)
          throw std::invalid_argument("Returning alternatives from a prefix is not supported "
                                      "in batch mode");
        if (return_attention)
          prefix_attention.resize(1);
        initialize_decoder_with_prefix(decoder,
                                       state,
                                       start_ids[0],
                                       prefix_ids->front(),
                                       return_attention ? &prefix_attention[0] : nullptr);
        start_ids[0] = prefix_ids->front().back();
        const dim_t prefix_length = prefix_ids->front().size();
        start_step += prefix_length;
        max_length = std::max(max_length - prefix_length, dim_t(0));
        min_length = std::max(min_length - prefix_length, dim_t(0));
      }

      // In this translation mode, we first expand the next "num_hypotheses" candidate words
      // before running the full decoding on each prefix. This is to ensure that we get unique
      // alternatives at this decoding position.
      expansion_results = BeamSearch(num_hypotheses).search(decoder,
                                                            state,
                                                            BestSampler(),
                                                            start_ids,
                                                            end_id,
                                                            start_step,
                                                            /*max_length=*/1,
                                                            /*min_length=*/1,
                                                            output_ids_map,
                                                            normalize_scores,
                                                            return_scores,
                                                            return_attention,
                                                            num_hypotheses);

      start_ids.resize(batch_size * num_hypotheses);
      for (size_t b = 0; b < batch_size; ++b) {
        auto& result = expansion_results[b];

        for (size_t i = 0; i < num_hypotheses; ++i) {
          // The next input is the words we just expanded.
          start_ids[b * num_hypotheses + i] = result.hypotheses[i].back();

          // Prepend expansion result with the prefix.
          if (prefix_ids) {
            result.hypotheses[i].insert(result.hypotheses[i].begin(),
                                        prefix_ids->at(b).begin(),
                                        prefix_ids->at(b).end());
            if (return_attention) {
              result.attention[i].insert(result.attention[i].begin(),
                                         prefix_attention[b].begin(),
                                         prefix_attention[b].end());
            }
          }
        }
      }

      start_step += 1;
      max_length = std::max(max_length - 1, dim_t(0));
      min_length = std::max(min_length - 1, dim_t(0));
    }

    auto results = search_strategy.search(decoder,
                                          state,
                                          sampler,
                                          start_ids,
                                          end_id,
                                          start_step,
                                          max_length,
                                          min_length,
                                          output_ids_map,
                                          normalize_scores,
                                          return_scores,
                                          return_attention,
                                          return_alternatives ? 1 : num_hypotheses,
                                          return_alternatives ? nullptr : prefix_ids);

    if (return_alternatives) {
      // Append to expansion results.
      for (size_t b = 0; b < batch_size; ++b) {
        auto& prefix = expansion_results[b];
        for (size_t i = 0; i < num_hypotheses; ++i) {
          auto& suffix = results[b * num_hypotheses + i];

          if (prefix.has_scores()) {
            if (normalize_scores) {
              const auto prefix_length = prefix.hypotheses[i].size();
              const auto suffix_length = suffix.hypotheses[0].size();
              prefix.scores[i] = (
                (prefix.scores[i] * prefix_length + suffix.scores[0] * suffix_length)
                / (prefix_length + suffix_length));
            } else {
              prefix.scores[i] += suffix.scores[0];
            }
          }

          prefix.hypotheses[i].insert(prefix.hypotheses[i].end(),
                                      std::make_move_iterator(suffix.hypotheses[0].begin()),
                                      std::make_move_iterator(suffix.hypotheses[0].end()));
          if (prefix.has_attention())
            prefix.attention[i].insert(prefix.attention[i].end(),
                                       std::make_move_iterator(suffix.attention[0].begin()),
                                       std::make_move_iterator(suffix.attention[0].end()));
        }
      }
      results = std::move(expansion_results);
    }

    // Remove EOS token.
    for (auto& result : results) {
      for (size_t i = 0; i < result.num_hypotheses(); ++i) {
        while (result.hypotheses[i].back() == end_id) {
          result.hypotheses[i].pop_back();
          if (result.has_attention())
            result.attention[i].pop_back();
        }
      }
    }

    return results;
  }

}
