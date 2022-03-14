#include "ctranslate2/decoding.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>

#include "ctranslate2/ops/ops.h"
#include "dispatch.h"

namespace ctranslate2 {

  static const ops::Gather gather;

  static inline void split_batch_beam(StorageView& input, dim_t beam_size) {
    Shape shape = input.shape();
    shape.insert(shape.begin() + 1, beam_size);
    shape[0] /= beam_size;
    input.reshape(std::move(shape));
  }

  static inline void merge_batch_beam(StorageView& input) {
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

  static void disable_token(StorageView& log_probs, const size_t id) {
    DEVICE_AND_TYPE_DISPATCH(log_probs.device(), log_probs.dtype(),
                             primitives<D>::strided_fill(log_probs.data<T>() + id,
                                                         static_cast<T>(-1e10),
                                                         log_probs.dim(-1),
                                                         log_probs.dim(0)));
  }

  static void penalize_previous_tokens(StorageView& log_probs,
                                       const StorageView& previous_ids,
                                       const float penalty) {
    StorageView previous_scores(log_probs.device(), log_probs.dtype());
    ops::Gather(/*axis=*/-1, /*batch_dims=*/1)(log_probs, previous_ids, previous_scores);

    DEVICE_AND_TYPE_DISPATCH(log_probs.device(), log_probs.dtype(),
                             primitives<D>::penalize_previous_tokens(log_probs.data<T>(),
                                                                     previous_scores.data<T>(),
                                                                     previous_ids.data<int32_t>(),
                                                                     static_cast<T>(penalty),
                                                                     log_probs.dim(0),
                                                                     previous_ids.dim(-1),
                                                                     log_probs.dim(-1)));
  }

  static void update_sample_with_prefix(const size_t step,
                                        StorageView& sampled_ids,
                                        StorageView& sampled_scores,
                                        const std::vector<std::vector<size_t>>& prefix_ids,
                                        const size_t end_id,
                                        const std::vector<dim_t>& batch_offset,
                                        StorageView* beam_origins = nullptr,
                                        const bool is_expanded = true) {
    const dim_t batch_size = sampled_scores.dim(0);
    const dim_t beam_size = sampled_scores.dim(1);
    for (dim_t i = 0; i < batch_size; ++i) {
      const auto& prefix = prefix_ids[batch_offset[i]];
      if (step > prefix.size())
        continue;

      for (dim_t k = 0; k < beam_size; ++k) {
        const dim_t flat_index = i * beam_size + k;
        auto& sampled_id = sampled_ids.at<int32_t>(flat_index);
        int32_t new_id = -1;
        float new_score = 0;

        // When step < prefix_length, we override the sampled ids with the prefix ids
        // and set the highest probability to the first beam.
        if (step < prefix.size()) {
          new_id = prefix[step];
          new_score = (k == 0 ? 0.f : float(-1e10));

        // When step == prefix_length (the first unconstrained decoding step),
        // only the first beam is expanded. It happens that </s> appears in the topk,
        // especially when k is large. This can produce incorrect and short predictions
        // that dominate others when no length normalization is used (see issue #277).
        // To mitigate this issue, we penalize </s> in secondary beams.
        } else if (k > 0 && static_cast<size_t>(sampled_id) == end_id) {
          new_id = 0;
          new_score = -1e10;
        }

        if (new_id >= 0) {
          sampled_id = new_id;
          TYPE_DISPATCH(sampled_scores.dtype(), sampled_scores.at<T>(flat_index) = T(new_score));
          if (beam_origins)
            beam_origins->at<int32_t>(flat_index) = (is_expanded ? i * beam_size : i);
        }
      }
    }
  }

  static inline void convert_to_original_word_ids(StorageView& ids,
                                                  const std::vector<size_t>& output_ids_map) {
    auto* ids_data = ids.data<int32_t>();
    for (dim_t i = 0; i < ids.size(); ++i)
      ids_data[i] = output_ids_map[ids_data[i]];
  }

  template <typename T>
  static void initialize_beam_scores(StorageView& scores,
                                     const dim_t batch_size,
                                     const dim_t beam_size) {
    const dim_t size = batch_size * beam_size;
    scores.resize({size});
    auto* data = scores.data<T>();
    for (dim_t i = 0; i < size; ++i) {
      data[i] = (i % beam_size == 0 ? T(0) : std::numeric_limits<T>::lowest());
    }
  }

  static StorageView unflatten_ids(StorageView& ids,
                                   const dim_t beam_size,
                                   const dim_t vocabulary_size,
                                   const bool is_expanded) {
    const dim_t num_ids = ids.size();
    StorageView beam_origins({num_ids}, DataType::INT32);

    auto* ids_data = ids.data<int32_t>();
    auto* origins_data = beam_origins.data<int32_t>();

    for (dim_t i = 0; i < num_ids; ++i) {
      const auto flat_id = ids_data[i];
      const auto beam_id = flat_id / vocabulary_size;
      const auto word_id = flat_id % vocabulary_size;
      const auto batch_id = i / ids.dim(-1);
      ids_data[i] = word_id;
      origins_data[i] = is_expanded ? batch_id * beam_size + beam_id : batch_id;
    }

    return beam_origins;
  }

  static void append_step_output(StorageView& history,    // [batch, beam, time, ...]
                                 StorageView step_output,  // [batch, beam, ...]
                                 const StorageView& beam_origins) {
    step_output.expand_dims(2);  // Insert time dimension.

    if (history) {
      const dim_t beam_size = history.dim(1);
      merge_batch_beam(history);
      gather(history, beam_origins);
      split_batch_beam(history, beam_size);
      const StorageView cur_history(std::move(history));
      ops::Concat(2)({&cur_history, &step_output}, history);
    } else {
      history = std::move(step_output);
    }
  }

  static std::vector<size_t> build_hypothesis(const StorageView& history,
                                              const dim_t batch,
                                              const dim_t beam) {
    const auto length = history.dim(-1);
    const auto* ids = history.index<int32_t>({batch, beam, 0});
    return std::vector<size_t>(ids, ids + length);
  }

  static std::vector<std::vector<float>> build_attention(const StorageView& history,
                                                         const dim_t batch,
                                                         const dim_t beam) {
    if (!history)
      return {};

    const auto source_length = history.dim(-1);
    const auto target_length = history.dim(-2);

    std::vector<std::vector<float>> attention;
    attention.reserve(target_length);
    for (dim_t t = 0; t < target_length; ++t) {
      const auto* vector = history.index<float>({batch, beam, t, 0});
      attention.emplace_back(vector, vector + source_length);
    }
    return attention;
  }

  static float compute_coverage_penalty(const std::vector<std::vector<float>>& attention,
                                        const float beta) {
    float penalty = 0;
    for (const auto& vector : attention) {
      const float coverage = std::accumulate(vector.begin(), vector.end(), 0.f);
      penalty += std::log(std::min(coverage, 1.f));
    }
    return beta * penalty;
  }

  static float finalize_hypothesis_score(float score,
                                         const bool normalize_score,
                                         const float length,
                                         const float length_penalty,
                                         const float coverage_penalty,
                                         const std::vector<std::vector<float>>& attention) {
    if (length_penalty != 0) {
      const float base = normalize_score ? length : (5.f + length) / 6.f;
      score /= std::pow(base, length_penalty);
    } else if (normalize_score) {
      score /= length;
    }

    if (coverage_penalty != 0)
      score += compute_coverage_penalty(attention, coverage_penalty);

    return score;
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

  static inline std::vector<std::vector<bool>>
  get_beams_divergence_from_prefix(const std::vector<std::vector<bool>>& beams_diverged_from_prefix,
                                   const size_t step,
                                   const StorageView& sampled_ids,
                                   const std::vector<std::vector<size_t>>& prefix_ids,
                                   const std::vector<dim_t>& batch_offset) {
    auto updated = beams_diverged_from_prefix;
    for (dim_t i = 0; i < sampled_ids.dim(0); ++i) {
      for (dim_t k = 0; k < sampled_ids.dim(1); ++k) {
        const size_t word_id = sampled_ids.at<int32_t>({i, k});
        const auto& prefix = prefix_ids[batch_offset[i]];
        updated[i][k] = (step >= prefix.size()
                         || beams_diverged_from_prefix[i][k]
                         || word_id != prefix[step]);
      }
    }
    return updated;
  }

  static inline bool
  all_beams_diverged_from_prefix(const std::vector<std::vector<bool>>& beams_diverged_from_prefix) {
    for (const auto& batch : beams_diverged_from_prefix) {
      for (const bool beam_diverged : batch) {
        if (!beam_diverged)
          return false;
      }
    }
    return true;
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
                     const size_t unk_id,
                     const dim_t start_step,
                     const dim_t max_length,
                     const dim_t min_length,
                     const std::vector<size_t>* output_ids_map,
                     const bool normalize_scores,
                     const bool return_scores,
                     const bool return_attention,
                     const size_t num_hypotheses,
                     const float repetition_penalty,
                     const bool disable_unk,
                     const std::vector<std::vector<size_t>>* prefix_ids) const {
    PROFILE("beam_search");
    const dim_t min_step = start_step + min_length;
    const dim_t max_step = start_step + max_length;
    const Device device = decoder.device();
    const DataType dtype = decoder.output_type();
    const bool expand_after_first_step = (device == Device::CPU);
    const dim_t batch_size = start_ids.size();

    StorageView topk_ids({batch_size}, DataType::INT32);
    StorageView topk_scores(dtype);

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
      TYPE_DISPATCH(dtype, initialize_beam_scores<T>(topk_scores, batch_size, _beam_size));
    }

    std::unique_ptr<BiasedDecoder> biased_decoder;
    std::vector<std::vector<bool>> beams_diverged_from_prefix;
    bool bias_towards_prefix = prefix_ids && _prefix_bias_beta > 0;
    if (bias_towards_prefix) {
      biased_decoder = std::make_unique<BiasedDecoder>();
      beams_diverged_from_prefix.resize(batch_size, std::vector<bool>(_beam_size, false));
    }
    const bool use_hard_prefix = prefix_ids && !bias_towards_prefix;

    StorageView logits(dtype, device);
    StorageView alive_seq(topk_ids.dtype());
    StorageView alive_attention;

    for (dim_t step = start_step; step < max_step; ++step) {
      const bool is_expanded = (!expand_after_first_step || step > start_step);

      // Compute log probs for the current step.
      StorageView attention_step(dtype, device);
      decoder(step,
              topk_ids.to(device),
              state,
              &logits,  // output shape: (cur_batch_size*beam_size x vocab_size), if not expanded beam_size is 1
              (return_attention || _coverage_penalty != 0) ? &attention_step : nullptr);

      const dim_t cur_batch_size = is_expanded ? logits.dim(0) / _beam_size : logits.dim(0);
      const dim_t vocabulary_size = logits.dim(-1);

      if (repetition_penalty != 1 && alive_seq) {
        merge_batch_beam(alive_seq);
        penalize_previous_tokens(logits, alive_seq.to(device), repetition_penalty);
        split_batch_beam(alive_seq, _beam_size);
      }

      StorageView log_probs(dtype, device);
      if (bias_towards_prefix) {
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

      // Prevent the generation of end_id until the minimum length is reached.
      if (step < min_step)
        disable_token(log_probs, end_id);
      if (disable_unk)
        disable_token(log_probs, unk_id);

      // Multiply by the current beam log probs.
      if (topk_scores) {
        DEVICE_AND_TYPE_DISPATCH(log_probs.device(), log_probs.dtype(),
                                 primitives<D>::add_depth_broadcast(topk_scores.to(device).data<T>(),
                                                                    log_probs.data<T>(),
                                                                    topk_scores.size(),
                                                                    log_probs.size()));
      }

      // Flatten the probs into a list of candidates.
      log_probs.reshape({cur_batch_size, -1});

      // TopK candidates.
      sampler(log_probs, topk_ids, topk_scores, _beam_size);

      // Unflatten the ids.
      StorageView gather_indices = unflatten_ids(topk_ids, _beam_size, vocabulary_size, is_expanded);

      if (output_ids_map)
        convert_to_original_word_ids(topk_ids, *output_ids_map);
      if (prefix_ids) {
        if (use_hard_prefix) {
          update_sample_with_prefix(step,
                                    topk_ids,
                                    topk_scores,
                                    *prefix_ids,
                                    end_id,
                                    batch_offset,
                                    &gather_indices,
                                    is_expanded);
        } else if (bias_towards_prefix) {
          beams_diverged_from_prefix = get_beams_divergence_from_prefix(beams_diverged_from_prefix,
                                                                        step,
                                                                        topk_ids,
                                                                        *prefix_ids,
                                                                        batch_offset);
        }
      }

      // Append last prediction.
      append_step_output(alive_seq, topk_ids, gather_indices);

      if (attention_step) {
        if (!is_expanded)
          expand_to_beam_size(attention_step, _beam_size);
        split_batch_beam(attention_step, _beam_size);
        append_step_output(alive_attention,
                           attention_step.to_float().to(Device::CPU),
                           gather_indices);
      }

      // Check if some hypotheses are finished.
      std::vector<int32_t> non_finished_index;
      non_finished_index.reserve(cur_batch_size);

      for (dim_t i = 0; i < cur_batch_size; ++i) {
        const dim_t batch_id = batch_offset[i];
        auto& result = results[batch_id];

        for (dim_t k = 0; k < _beam_size; ++k) {
          const size_t last_id = topk_ids.at<int32_t>({i, k});

          if (last_id == end_id || step + 1 == max_step) {
            if (k == 0)
              top_beam_finished[i] = true;

            // Build the hypothesis and compute its score.
            auto hypothesis = build_hypothesis(alive_seq, i, k);
            auto attention = build_attention(alive_attention, i, k);
            auto score = finalize_hypothesis_score(topk_scores.scalar_at<float>({i, k}),
                                                   normalize_scores,
                                                   hypothesis.size(),
                                                   _length_penalty,
                                                   _coverage_penalty,
                                                   attention);

            // Register this hypothesis.
            result.scores.emplace_back(score);
            result.hypotheses.emplace_back(std::move(hypothesis));
            if (return_attention)
              result.attention.emplace_back(std::move(attention));

            // Prevent this beam from advancing in the next step.
            TYPE_DISPATCH(dtype, topk_scores.at<T>({i, k}) = T(-1e10));
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
        batch_offset = index_vector(batch_offset, non_finished_index);
        top_beam_finished = index_vector(top_beam_finished, non_finished_index);
        if (bias_towards_prefix)
          beams_diverged_from_prefix = index_vector(beams_diverged_from_prefix, non_finished_index);

        StorageView keep_batches({next_batch_size}, non_finished_index);
        gather(topk_ids, keep_batches);
        gather(topk_scores, keep_batches);
        gather(alive_seq, keep_batches);
        if (alive_attention)
          gather(alive_attention, keep_batches);

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

      topk_ids.reshape({next_batch_size * _beam_size});
      topk_scores.reshape({next_batch_size * _beam_size});

      if (bias_towards_prefix)
        bias_towards_prefix = !all_beams_diverged_from_prefix(beams_diverged_from_prefix);
    }

    return results;
  }

  std::vector<GenerationResult<size_t>>
  GreedySearch::search(layers::Decoder& decoder,
                       layers::DecoderState& state,
                       const Sampler& sampler,
                       const std::vector<size_t>& start_ids,
                       const size_t end_id,
                       const size_t unk_id,
                       const dim_t start_step,
                       const dim_t max_length,
                       const dim_t min_length,
                       const std::vector<size_t>* output_ids_map,
                       const bool normalize_scores,
                       const bool return_scores,
                       const bool return_attention,
                       const size_t,
                       const float repetition_penalty,
                       const bool disable_unk,
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
    StorageView alive_seq(DataType::INT32);
    StorageView attention_step;
    StorageView attention_step_device(dtype, device);

    for (dim_t step = start_step; step < max_step; ++step) {
      decoder(step,
              sample_from.to(device),
              state,
              &logits,
              return_attention ? &attention_step_device : nullptr);

      if (repetition_penalty != 1 && alive_seq)
        penalize_previous_tokens(logits, alive_seq.to(device), repetition_penalty);

      // Compute log probs only if scores should be returned.
      StorageView log_probs(dtype, device);
      if (return_scores)
        ops::LogSoftMax()(logits);
      log_probs.shallow_copy(logits);

      // Prevent the generation of end_id until the minimum length is reached.
      if (step < min_step)
        disable_token(log_probs, end_id);
      if (disable_unk)
        disable_token(log_probs, unk_id);

      sampler(log_probs, best_ids, best_probs);
      if (output_ids_map)
        convert_to_original_word_ids(best_ids, *output_ids_map);
      if (prefix_ids)
        update_sample_with_prefix(step, best_ids, best_probs, *prefix_ids, end_id, batch_offset);
      if (return_attention)
        attention_step.copy_from(attention_step_device.to_float());

      // When repetition penalty is enabled, we should keep the previously generated tokens.
      if (repetition_penalty != 1) {
        if (alive_seq) {
          const StorageView cur_alive_seq = std::move(alive_seq);
          ops::Concat(-1)({&cur_alive_seq, &best_ids}, alive_seq);
        } else {
          alive_seq = best_ids;
        }
      }

      const dim_t cur_batch_size = log_probs.dim(0);
      std::vector<int32_t> non_finished_index;
      non_finished_index.reserve(cur_batch_size);

      for (dim_t i = 0; i < cur_batch_size; ++i) {
        const size_t word_id = best_ids.at<int32_t>(i);
        const size_t batch_id = batch_offset[i];
        results[batch_id].hypotheses[0].push_back(word_id);
        if (return_scores)
          results[batch_id].scores[0] += best_probs.scalar_at<float>({i, 0});
        if (return_attention) {
          const auto* attn = attention_step.index<float>({i, 0});
          results[batch_id].attention[0].emplace_back(attn, attn + attention_step.dim(-1));
        }
        if (word_id != end_id) {
          non_finished_index.emplace_back(i);
          sample_from.at<int32_t>(i) = word_id;
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
        if (alive_seq)
          gather(alive_seq, alive);
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

  static layers::DecoderState get_batch_state(const layers::DecoderState& state,
                                              const int32_t batch_id) {
    const Device device = state.begin()->second.device();
    const ops::Gather gather_op;

    StorageView indices(batch_id, device);
    indices.reshape({1});

    layers::DecoderState batch_state;
    batch_state.reserve(state.size());

    for (const auto& pair : state) {
      const auto& name = pair.first;
      const auto& value = pair.second;
      StorageView batch_value(value.dtype(), device);
      if (value)
        gather_op(value, indices, batch_value);
      batch_state.emplace(name, std::move(batch_value));
    }

    return batch_state;
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
         const size_t unk_id,
         dim_t max_length,
         dim_t min_length,
         const size_t num_hypotheses,
         const bool return_alternatives,
         const bool return_scores,
         const bool return_attention,
         const bool normalize_scores,
         const float repetition_penalty,
         const bool disable_unk) {
    const size_t batch_size = start_ids.size();

    if (return_alternatives && batch_size > 1) {
      // return_alternatives mode currently does not support batch decoding.
      std::vector<GenerationResult<size_t>> results;
      results.reserve(batch_size);
      for (size_t i = 0; i < batch_size; ++i) {
        layers::DecoderState batch_state = get_batch_state(state, i);
        std::vector<size_t> batch_start_ids{start_ids[i]};
        std::vector<std::vector<size_t>> batch_prefix_ids;
        if (prefix_ids)
          batch_prefix_ids.emplace_back(prefix_ids->at(i));
        results.emplace_back(decode(decoder,
                                    batch_state,
                                    search_strategy,
                                    sampler,
                                    batch_start_ids,
                                    batch_prefix_ids.empty() ? nullptr : &batch_prefix_ids,
                                    output_ids_map,
                                    end_id,
                                    unk_id,
                                    max_length,
                                    min_length,
                                    num_hypotheses,
                                    return_alternatives,
                                    return_scores,
                                    return_attention,
                                    normalize_scores,
                                    repetition_penalty,
                                    disable_unk)[0]);
      }
      return results;
    }

    dim_t start_step = 0;

    std::vector<GenerationResult<size_t>> expansion_results;
    if (return_alternatives) {
      std::vector<std::vector<std::vector<float>>> prefix_attention;
      if (prefix_ids) {
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
                                                            unk_id,
                                                            start_step,
                                                            /*max_length=*/1,
                                                            /*min_length=*/1,
                                                            output_ids_map,
                                                            normalize_scores,
                                                            return_scores,
                                                            return_attention,
                                                            num_hypotheses,
                                                            /*repetition_penalty=*/1,
                                                            disable_unk);

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
                                          unk_id,
                                          start_step,
                                          max_length,
                                          min_length,
                                          output_ids_map,
                                          normalize_scores,
                                          return_scores,
                                          return_attention,
                                          return_alternatives ? 1 : num_hypotheses,
                                          repetition_penalty,
                                          disable_unk,
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
