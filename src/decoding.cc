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

  static void gather_beam_flat(StorageView& data, const StorageView& indices, dim_t beam_size) {
    merge_batch_beam(data);
    gather(data, indices);
    split_batch_beam(data, beam_size);
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

  static void update_sample_with_prefix(const size_t step,
                                        StorageView& sampled_ids,
                                        StorageView& sampled_scores,
                                        const std::vector<std::vector<size_t>>& prefix_ids,
                                        const size_t end_id,
                                        const std::vector<dim_t>& batch_offset,
                                        const dim_t beam_size = 1,
                                        StorageView* beam_origins = nullptr,
                                        const bool is_expanded = true) {
    const dim_t batch_size = sampled_scores.dim(0);
    for (dim_t i = 0; i < batch_size; ++i) {
      const auto& prefix = prefix_ids[batch_offset[i]];
      if (step > prefix.size())
        continue;

      const dim_t num_samples = sampled_scores.dim(1);
      for (dim_t k = 0; k < num_samples; ++k) {
        const dim_t flat_index = i * num_samples + k;
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

  static inline void convert_to_original_word_ids(const layers::Decoder& decoder,
                                                  StorageView& ids) {
    if (!decoder.output_layer_is_updated())
      return;
    auto* ids_data = ids.data<int32_t>();
    for (dim_t i = 0; i < ids.size(); ++i)
      ids_data[i] = decoder.to_original_word_id(ids_data[i]);
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
                                 const StorageView* beam_origins = nullptr) {
    step_output.expand_dims(2);  // Insert time dimension.

    if (history) {
      if (beam_origins)
        gather_beam_flat(history, *beam_origins, step_output.dim(1));
      const StorageView cur_history(std::move(history));
      ops::Concat(2)({&cur_history, &step_output}, history);
    } else {
      history = std::move(step_output);
    }
  }

  static std::vector<size_t> build_hypothesis(const StorageView& history,
                                              const dim_t batch,
                                              const dim_t beam,
                                              const bool ignore_last) {
    const auto length = history.dim(-1) - dim_t(ignore_last);
    const auto* ids = history.index<int32_t>({batch, beam, 0});
    return std::vector<size_t>(ids, ids + length);
  }

  static std::vector<std::vector<float>> build_attention(const StorageView& history,
                                                         const dim_t batch,
                                                         const dim_t beam,
                                                         const bool ignore_last) {
    if (!history)
      return {};

    const auto source_length = history.dim(-1);
    const auto target_length = history.dim(-2) - dim_t(ignore_last);

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
    for (size_t column = 0; column < attention[0].size(); column++) {
      float coverage = 0;
      for (size_t row = 0; row < attention.size(); row++)
        coverage += attention[row][column];
      if (coverage > 0)
        penalty += std::log(std::min(coverage, 1.f));
    }
    return beta * penalty;
  }

  static float finalize_hypothesis_score(float score,
                                         const float length,
                                         const float length_penalty,
                                         const float coverage_penalty,
                                         const std::vector<std::vector<float>>* attention) {
    score /= std::pow(length, length_penalty);

    if (coverage_penalty != 0) {
      if (!attention)
        throw std::runtime_error("The attention weights are required to apply the coverage penalty");
      score += compute_coverage_penalty(*attention, coverage_penalty);
    }

    return score;
  }

  // Sort hypotheses from best to worst score, in the limit of max_hypotheses.
  static inline void sort_hypotheses(DecodingResult& result,
                                     size_t max_hypotheses,
                                     bool keep_scores,
                                     bool keep_attention) {
    std::vector<size_t> idx(result.hypotheses.size());
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

    if (keep_attention)
      result.attention = index_vector(result.attention, idx);
    else
      result.attention.clear();
  }

  static inline void finalize_result(DecodingResult& result,
                                     const size_t max_hypotheses,
                                     const float length_penalty,
                                     const float coverage_penalty,
                                     const bool keep_scores,
                                     const bool keep_attention) {
    for (size_t i = 0; i < result.scores.size(); ++i) {
      const auto* attention = result.attention.empty() ? nullptr : &result.attention[i];
      result.scores[i] = finalize_hypothesis_score(result.scores[i],
                                                   result.hypotheses[i].size(),
                                                   length_penalty,
                                                   coverage_penalty,
                                                   attention);
    }

    sort_hypotheses(result, max_hypotheses, keep_scores, keep_attention);
  }

  BiasedDecoder::BiasedDecoder(const float prefix_bias_beta,
                               const std::vector<std::vector<size_t>>& prefix_ids)
    : _prefix_bias_beta(prefix_bias_beta)
    , _prefix_ids(prefix_ids)
  {
  }

  void BiasedDecoder::decode(const dim_t cur_batch_size,
                             const size_t step,
                             const std::vector<dim_t>& batch_offset,
                             const std::vector<std::vector<bool>>& beams_diverged_from_prefix,
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
    StorageView scalar_discount(1 - _prefix_bias_beta, Device::CPU);
    assert (num_beams % cur_batch_size == 0);
    const dim_t cur_beam_size = num_beams / cur_batch_size;
    for (dim_t b = 0; b < num_beams; ++b) {
      StorageView &logit_beam = *(logit_beam_views[b]);
      StorageView &log_prob_beam = *(log_prob_beam_views[b]);
      const dim_t index_batch = b / cur_beam_size;
      const dim_t index_beam = b % cur_beam_size;
      const auto& prefix = _prefix_ids[batch_offset[index_batch]];
      if (static_cast<size_t>(step) < prefix.size()
          && !beams_diverged_from_prefix[index_batch][index_beam]) {
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
          beta_scalar = StorageView(static_cast<T>(_prefix_bias_beta), Device::CPU));
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
    for (dim_t i = 0; i < dim_t(updated.size()); ++i) {
      for (dim_t k = 0; k < dim_t(updated[i].size()); ++k) {
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

  static inline size_t get_max_candidates(const dim_t beam_size, const float patience) {
    return std::round(float(beam_size) * patience);
  }


  BeamSearch::BeamSearch(const dim_t beam_size,
                         const float length_penalty,
                         const float coverage_penalty,
                         const float prefix_bias_beta,
                         const float patience)
    : _beam_size(beam_size)
    , _length_penalty(length_penalty)
    , _coverage_penalty(coverage_penalty)
    , _prefix_bias_beta(prefix_bias_beta)
    , _max_candidates(get_max_candidates(beam_size, patience))
  {
  }

  std::vector<DecodingResult>
  BeamSearch::search(layers::Decoder& decoder,
                     layers::DecoderState& state,
                     const Sampler& sampler,
                     const std::vector<size_t>& start_ids,
                     const size_t end_id,
                     const dim_t start_step,
                     const dim_t max_length,
                     const dim_t min_length,
                     const bool return_scores,
                     const bool return_attention,
                     const size_t num_hypotheses,
                     const bool include_eos_in_scores,
                     const bool include_eos_in_hypotheses,
                     const std::vector<std::shared_ptr<LogitsProcessor>>& logits_processors,
                     const std::vector<std::vector<size_t>>* prefix_ids) const {
    PROFILE("beam_search");
    const Device device = decoder.device();
    const DataType dtype = decoder.output_type();
    const dim_t vocabulary_size = decoder.output_size();
    const dim_t batch_size = start_ids.size();

    // We get more candidates than the beam size so that if half the candidates are EOS,
    // we can replace finished hypotheses with active beams.
    const dim_t num_candidates = _beam_size * 2;

    // Only the first beam is considered in the first step. As an additional optimization
    // we try to run the first step without expanding the batch size.
    const bool expand_after_first_step = (device == Device::CPU
                                          && num_candidates <= vocabulary_size);

    // We can exit early when the first beam finishes and no penalties are used.
    const bool allow_early_exit = (_length_penalty == 0 && _coverage_penalty == 0);

    StorageView topk_ids({batch_size}, DataType::INT32);
    StorageView topk_scores(dtype);

    std::vector<bool> top_beam_finished(batch_size, false);
    std::vector<dim_t> batch_offset(batch_size);
    std::vector<DecodingResult> results(batch_size);
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
      biased_decoder = std::make_unique<BiasedDecoder>(_prefix_bias_beta, *prefix_ids);
      beams_diverged_from_prefix.resize(batch_size, std::vector<bool>(_beam_size, false));
    }
    const bool use_hard_prefix = prefix_ids && !bias_towards_prefix;

    StorageView logits(dtype, device);
    StorageView alive_seq(topk_ids.dtype());
    StorageView alive_attention;

    for (dim_t step = 0; step < max_length; ++step) {
      const bool is_expanded = (!expand_after_first_step || step > 0);

      // Compute log probs for the current step.
      StorageView attention_step(dtype, device);
      convert_to_original_word_ids(decoder, topk_ids);
      decoder(start_step + step,
              topk_ids.to(device),
              state,
              &logits,  // output shape: (cur_batch_size*beam_size x vocab_size), if not expanded beam_size is 1
              (return_attention || _coverage_penalty != 0) ? &attention_step : nullptr);

      const dim_t cur_batch_size = is_expanded ? logits.dim(0) / _beam_size : logits.dim(0);

      DisableTokens disable_tokens(logits);

      // Prevent the generation of end_id until the minimum length is reached.
      if (step < min_length)
        disable_tokens.add(end_id);

      if (!logits_processors.empty()) {
        if (alive_seq)
          merge_batch_beam(alive_seq);
        for (const auto& logits_processor : logits_processors)
          logits_processor->apply(step, logits, disable_tokens, alive_seq, batch_offset, prefix_ids);
        if (alive_seq)
          split_batch_beam(alive_seq, _beam_size);
      }

      disable_tokens.apply();

      StorageView log_probs(dtype, device);
      if (bias_towards_prefix) {
        biased_decoder->decode(cur_batch_size,
                               step,
                               batch_offset,
                               beams_diverged_from_prefix,
                               logits,
                               log_probs);
      } else {
        ops::LogSoftMax()(logits);
        log_probs.shallow_copy(logits);
      }

      // Multiply by the current beam log probs.
      StorageView topk_scores_prev(dtype);
      if (topk_scores) {
        DEVICE_AND_TYPE_DISPATCH(log_probs.device(), log_probs.dtype(),
                                 primitives<D>::add_depth_broadcast(topk_scores.to(device).data<T>(),
                                                                    log_probs.data<T>(),
                                                                    topk_scores.size(),
                                                                    log_probs.size()));

        if (!include_eos_in_scores) {
          topk_scores_prev = topk_scores;
          topk_scores_prev.reshape({cur_batch_size, _beam_size});
        }
      }

      // Flatten the probs into a list of candidates.
      log_probs.reshape({cur_batch_size, -1});

      // TopK candidates.
      sampler(log_probs, topk_ids, topk_scores, num_candidates);

      // Unflatten the ids.
      StorageView gather_indices = unflatten_ids(topk_ids, _beam_size, vocabulary_size, is_expanded);

      if (prefix_ids) {
        if (use_hard_prefix) {
          update_sample_with_prefix(step,
                                    topk_ids,
                                    topk_scores,
                                    *prefix_ids,
                                    end_id,
                                    batch_offset,
                                    _beam_size,
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
      append_step_output(alive_seq, topk_ids, &gather_indices);

      if (attention_step) {
        if (!is_expanded)
          expand_to_beam_size(attention_step, _beam_size);
        split_batch_beam(attention_step, _beam_size);
        append_step_output(alive_attention, attention_step.to_float().to(Device::CPU));
        gather_beam_flat(alive_attention, gather_indices, num_candidates);
      }

      // Check if some hypotheses are finished.
      std::vector<int32_t> non_finished_index;
      non_finished_index.reserve(cur_batch_size);

      // Only keep the first beam_size candidates.
      StorageView active_beams({cur_batch_size * _beam_size}, DataType::INT32);

      for (dim_t i = 0; i < cur_batch_size; ++i) {
        const dim_t batch_id = batch_offset[i];
        auto& result = results[batch_id];
        dim_t secondary_candidates_offset = _beam_size;

        for (dim_t k = 0; k < _beam_size; ++k) {
          const size_t last_id = topk_ids.at<int32_t>({i, k});
          const dim_t prefix_length = use_hard_prefix ? prefix_ids->at(batch_id).size() : 0;
          dim_t next_beam_id = k;

          if ((last_id == end_id && step >= prefix_length) || step + 1 == max_length) {
            if (k == 0)
              top_beam_finished[i] = true;

            bool ignore_last_score = false;
            bool ignore_last_token = false;
            if (last_id == end_id) {
              ignore_last_score = !include_eos_in_scores;
              ignore_last_token = !include_eos_in_hypotheses;
            }

            // Register this hypothesis.
            const StorageView& scores = ignore_last_score ? topk_scores_prev : topk_scores;
            result.scores.emplace_back(scores.scalar_at<float>({i, k}));
            result.hypotheses.emplace_back(build_hypothesis(alive_seq, i, k, ignore_last_token));
            if (alive_attention)
              result.attention.emplace_back(build_attention(alive_attention, i, k, ignore_last_token));

            // Move another active beam to this position.
            for (dim_t j = secondary_candidates_offset; j < num_candidates; ++j) {
              const auto candidate = topk_ids.at<int32_t>({i, j});
              if (static_cast<size_t>(candidate) != end_id) {
                next_beam_id = j;
                secondary_candidates_offset = j + 1;
                break;
              }
            }
          }

          active_beams.at<int32_t>(i * _beam_size + k) = i * num_candidates + next_beam_id;
        }

        bool is_finished = false;
        if (step + 1 == max_length)
          is_finished = true;
        else if (allow_early_exit)
          is_finished = top_beam_finished[i] && result.hypotheses.size() >= num_hypotheses;
        else
          is_finished = result.hypotheses.size() >= _max_candidates;

        if (is_finished) {
          finalize_result(result,
                          num_hypotheses,
                          _length_penalty,
                          _coverage_penalty,
                          return_scores,
                          return_attention);
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

      gather(gather_indices, active_beams);
      gather_beam_flat(topk_ids, active_beams, _beam_size);
      gather_beam_flat(topk_scores, active_beams, _beam_size);
      gather_beam_flat(alive_seq, active_beams, _beam_size);
      if (alive_attention)
        gather_beam_flat(alive_attention, active_beams, _beam_size);

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


  GreedySearch::GreedySearch(const float length_penalty, const float coverage_penalty)
    : _length_penalty(length_penalty)
    , _coverage_penalty(coverage_penalty)
  {
  }

  std::vector<DecodingResult>
  GreedySearch::search(layers::Decoder& decoder,
                       layers::DecoderState& state,
                       const Sampler& sampler,
                       const std::vector<size_t>& start_ids,
                       const size_t end_id,
                       const dim_t start_step,
                       const dim_t max_length,
                       const dim_t min_length,
                       const bool return_scores,
                       const bool return_attention,
                       const size_t num_hypotheses,
                       const bool include_eos_in_scores,
                       const bool include_eos_in_hypotheses,
                       const std::vector<std::shared_ptr<LogitsProcessor>>& logits_processors,
                       const std::vector<std::vector<size_t>>* prefix_ids) const {
    const dim_t batch_size = start_ids.size();

    // We can return multiple hypotheses from greedy search when random sampling is enabled.
    // In that case we replicate the batches and then merge the hypotheses in a single result.
    if (num_hypotheses > 1) {
      expand_to_beam_size(state, num_hypotheses);

      std::vector<size_t> repeat_start_ids = repeat_vector(start_ids, num_hypotheses);
      std::vector<std::vector<size_t>> repeat_prefix_ids;
      if (prefix_ids)
        repeat_prefix_ids = repeat_vector(*prefix_ids, num_hypotheses);

      std::vector<DecodingResult> results = search(decoder,
                                                   state,
                                                   sampler,
                                                   repeat_start_ids,
                                                   end_id,
                                                   start_step,
                                                   max_length,
                                                   min_length,
                                                   /*return_scores=*/true,
                                                   return_attention,
                                                   /*num_hypotheses=*/1,
                                                   include_eos_in_scores,
                                                   include_eos_in_hypotheses,
                                                   logits_processors,
                                                   prefix_ids ? &repeat_prefix_ids : nullptr);

      std::vector<DecodingResult> final_results(batch_size);

      for (size_t i = 0; i < results.size(); ++i) {
        auto& result = results[i];
        auto& final_result = final_results[i / num_hypotheses];

        final_result.hypotheses.emplace_back(std::move(result.hypotheses[0]));
        final_result.scores.emplace_back(result.scores[0]);
        if (return_attention)
          final_result.attention.emplace_back(std::move(result.attention[0]));
      }

      for (auto& result : final_results)
        sort_hypotheses(result, num_hypotheses, return_scores, return_attention);

      return final_results;
    }

    PROFILE("greedy_search");
    const Device device = decoder.device();
    const DataType dtype = decoder.output_type();
    const bool gather_attention = (return_attention || (return_scores && _coverage_penalty != 0));

    StorageView sample_from({batch_size}, DataType::INT32);

    StorageView logits(dtype, device);
    std::vector<dim_t> batch_offset(batch_size);
    std::vector<DecodingResult> results(batch_size);
    for (dim_t i = 0; i < batch_size; ++i) {
      batch_offset[i] = i;
      sample_from.at<int32_t>(i) = start_ids[i];
      results[i].hypotheses.resize(1);
      if (return_scores)
        results[i].scores.resize(1, 0.f);
      if (return_attention)
        results[i].attention.resize(1);
    }

    StorageView best_ids(DataType::INT32);
    StorageView best_probs(dtype);
    StorageView alive_seq(DataType::INT32);
    StorageView attention_step;
    StorageView attention_step_device(dtype, device);

    for (dim_t step = 0; step < max_length; ++step) {
      convert_to_original_word_ids(decoder, sample_from);
      decoder(start_step + step,
              sample_from.to(device),
              state,
              &logits,
              gather_attention ? &attention_step_device : nullptr);

      DisableTokens disable_tokens(logits);

      // Prevent the generation of end_id until the minimum length is reached.
      if (step < min_length)
        disable_tokens.add(end_id);

      for (const auto& logits_processor : logits_processors)
        logits_processor->apply(step, logits, disable_tokens, alive_seq, batch_offset, prefix_ids);

      disable_tokens.apply();

      // Compute log probs only if required.
      StorageView log_probs(dtype, device);
      if (return_scores)
        ops::LogSoftMax()(logits);
      log_probs.shallow_copy(logits);

      sampler(log_probs, best_ids, best_probs);
      if (prefix_ids)
        update_sample_with_prefix(step, best_ids, best_probs, *prefix_ids, end_id, batch_offset);
      if (attention_step_device)
        attention_step.copy_from(attention_step_device.to_float());

      if (!logits_processors.empty()) {
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
        const dim_t prefix_length = prefix_ids ? prefix_ids->at(batch_id).size() : 0;

        if (word_id != end_id || include_eos_in_hypotheses) {
          results[batch_id].hypotheses[0].push_back(word_id);
          if (attention_step) {
            const auto* attn = attention_step.index<float>({i, 0});
            results[batch_id].attention[0].emplace_back(attn, attn + attention_step.dim(-1));
          }
        }

        if (word_id != end_id || include_eos_in_scores) {
          if (return_scores)
            results[batch_id].scores[0] += best_probs.scalar_at<float>({i, 0});
        }

        const bool is_finished = ((word_id == end_id && step >= prefix_length)
                                  || (step + 1 == max_length));

        if (is_finished) {
          finalize_result(results[batch_id],
                          1,
                          _length_penalty,
                          _coverage_penalty,
                          return_scores,
                          return_attention);
        } else {
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

    return results;
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

  static std::pair<std::vector<size_t>, std::vector<std::vector<size_t>>>
  split_start_tokens(const std::vector<std::vector<size_t>>& start_tokens) {
    std::vector<size_t> start_ids;
    std::vector<std::vector<size_t>> prefix_ids;
    start_ids.reserve(start_tokens.size());
    prefix_ids.reserve(start_tokens.size());
    bool only_start_token = true;

    for (const auto& tokens : start_tokens) {
      if (tokens.empty())
        throw std::invalid_argument("One input has no decoder start token");
      if (tokens.size() > 1)
        only_start_token = false;

      start_ids.emplace_back(tokens.front());
      prefix_ids.emplace_back(tokens.begin() + 1, tokens.end());
    }

    if (only_start_token)
      prefix_ids.clear();

    return std::make_pair(std::move(start_ids), std::move(prefix_ids));
  }

  static void validate_decoding_options(const DecodingOptions& options) {
    if (options.beam_size == 0)
      throw std::invalid_argument("The beam size must be > 0");
    if (options.patience <= 0)
      throw std::invalid_argument("The patience factor must be > 0");
    if (options.num_hypotheses == 0)
      throw std::invalid_argument("The number of hypotheses must be > 0");
    if (options.num_hypotheses > get_max_candidates(options.beam_size, options.patience)
        && !options.return_alternatives
        && !(options.beam_size == 1 && options.sampling_topk != 1))
      throw std::invalid_argument("The number of hypotheses cannot be greater than "
                                  "beam_size * patience");
    if (options.min_length > options.max_length)
      throw std::invalid_argument("The minimum decoding length is greater than "
                                  "the maximum decoding length");
    if (options.max_length == 0)
      throw std::invalid_argument("The maximum decoding length must be > 0");
    if (options.repetition_penalty <= 0)
      throw std::invalid_argument("The repetition penalty must be > 0");
    if (options.prefix_bias_beta >= 1)
      throw std::invalid_argument("The beta value in biased decoding must be < 1");
    if (options.prefix_bias_beta > 0 && options.return_alternatives)
      throw std::invalid_argument("Biased decoding is not compatible with the return_alternatives "
                                  "mode");
    if (options.return_alternatives
        && (options.min_alternative_expansion_prob < 0
            || options.min_alternative_expansion_prob > 1))
      throw std::invalid_argument("The minimum alternative expansion probability must be "
                                  "between 0 and 1");
  }

  static std::unique_ptr<const Sampler>
  make_sampler(const DecodingOptions& options) {
    if (options.sampling_topk == 1)
      return std::make_unique<BestSampler>();
    else
      return std::make_unique<RandomSampler>(options.sampling_topk, options.sampling_temperature);
  }

  static std::unique_ptr<const SearchStrategy>
  make_search_strategy(const DecodingOptions& options) {
    if (options.beam_size == 1 && options.prefix_bias_beta == 0)
      return std::make_unique<GreedySearch>(options.length_penalty, options.coverage_penalty);
    else
      return std::make_unique<BeamSearch>(options.beam_size,
                                          options.length_penalty,
                                          options.coverage_penalty,
                                          options.prefix_bias_beta,
                                          options.patience);
  }

  static std::vector<std::shared_ptr<LogitsProcessor>>
  make_logits_processors(const DecodingOptions& options) {
    std::vector<std::shared_ptr<LogitsProcessor>> processors;

    if (options.repetition_penalty != 1)
      processors.emplace_back(std::make_shared<RepetitionPenalty>(options.repetition_penalty));

    if (options.no_repeat_ngram_size > 0)
      processors.emplace_back(std::make_shared<NoRepeatNgram>(options.no_repeat_ngram_size));

    if (!options.disable_ids.empty())
      processors.emplace_back(std::make_shared<SuppressTokens>(options.disable_ids));

    if (!options.disable_ids_begin.empty())
      processors.emplace_back(std::make_shared<SuppressTokensBegin>(options.disable_ids_begin));

    if (!options.disable_sequences.empty())
      processors.emplace_back(std::make_shared<SuppressSequences>(options.disable_sequences));

    for (const auto& processor : options.logits_processors)
      processors.emplace_back(processor);

    return processors;
  }

  static DecodingResult
  decode_alternatives(layers::Decoder& decoder,
                      layers::DecoderState& state,
                      std::vector<size_t> start_tokens,
                      const size_t end_id,
                      const DecodingOptions& options) {
    DecodingResult result;
    result.hypotheses.resize(options.num_hypotheses);
    if (options.return_scores)
      result.scores.resize(options.num_hypotheses, 0);
    if (options.return_attention)
      result.attention.resize(options.num_hypotheses);

    if (start_tokens.empty())
      throw std::invalid_argument("One input has no decoder start token");
    if (start_tokens.size() > options.max_length + 1)
      start_tokens.resize(options.max_length + 1);

    const dim_t min_length = options.min_length;
    const dim_t max_length = options.max_length;
    const dim_t prefix_length = start_tokens.size() - 1;
    dim_t start_step = options.start_step;

    if (prefix_length > 0) {
      // Initialize the decoder state with the prefix.
      const Device device = decoder.device();
      StorageView attention(decoder.output_type(), device);
      StorageView input_ids({1, prefix_length},
                            std::vector<int32_t>(start_tokens.begin(),
                                                 start_tokens.begin() + prefix_length),
                            device);

      convert_to_original_word_ids(decoder, input_ids);
      decoder(start_step,
              input_ids,
              state,
              /*logits=*/nullptr,
              options.return_attention ? &attention : nullptr);

      for (size_t i = 0; i < options.num_hypotheses; ++i) {
        result.hypotheses[i] = std::vector<size_t>(start_tokens.begin() + 1, start_tokens.end());

        if (options.return_attention) {
          if (attention.device() != Device::CPU)
            attention = attention.to_float().to(Device::CPU);
          for (dim_t t = 0; t < prefix_length; ++t) {
            const float* vector = attention.index<float>({0, t, 0});
            result.attention[i].emplace_back(vector, vector + attention.dim(-1));
          }
        }
      }

      if (prefix_length == max_length)
        return result;

      start_step += prefix_length;
    }

    std::vector<size_t> start_ids{start_tokens.back()};

    const auto logits_processors = make_logits_processors(options);

    // Expand the next "num_hypotheses" candidate words using the beam search.
    BeamSearch beam(options.num_hypotheses);
    DecodingResult expansion_result = beam.search(decoder,
                                                  state,
                                                  BestSampler(),
                                                  start_ids,
                                                  end_id,
                                                  start_step,
                                                  /*max_length=*/1,
                                                  /*min_length=*/1,
                                                  /*return_scores=*/true,
                                                  options.return_attention,
                                                  options.num_hypotheses,
                                                  options.include_eos_in_scores,
                                                  options.include_eos_in_hypotheses,
                                                  logits_processors)[0];

    start_ids.clear();

    for (size_t i = 0; i < options.num_hypotheses; ++i) {
      const float prob = std::exp(expansion_result.scores[i]);
      if (prob < options.min_alternative_expansion_prob)
        break;

      // Add expanded word to the result.
      result.hypotheses[i].emplace_back(expansion_result.hypotheses[i].back());
      if (options.return_attention)
        result.attention[i].emplace_back(std::move(expansion_result.attention[i].back()));
      if (options.return_scores)
        result.scores[i] = expansion_result.scores[i];

      // The next input is the words we just expanded.
      start_ids.push_back(result.hypotheses[i].back());
    }

    if (start_ids.size() < options.num_hypotheses) {
      // Reduce state to the effective number of alternatives.
      const dim_t num_alternatives = start_ids.size();
      for (auto& pair : state)
        pair.second.resize(0, num_alternatives);

      result.hypotheses.resize(num_alternatives);
      if (options.return_scores)
        result.scores.resize(num_alternatives);
      if (options.return_attention)
        result.attention.resize(num_alternatives);
    }

    start_step += 1;
    if (start_step == max_length)
      return result;

    // Continue the decoding from each alternative words independently.
    const auto search_strategy = make_search_strategy(options);
    const auto sampler = make_sampler(options);
    auto suffix_results = search_strategy->search(decoder,
                                                  state,
                                                  *sampler,
                                                  start_ids,
                                                  end_id,
                                                  start_step,
                                                  std::max(max_length - start_step, dim_t(0)),
                                                  std::max(min_length - start_step, dim_t(0)),
                                                  options.return_scores,
                                                  options.return_attention,
                                                  /*num_hypotheses=*/1,
                                                  options.include_eos_in_scores,
                                                  options.include_eos_in_hypotheses,
                                                  logits_processors);

    // Update the result with the suffix decoding.
    for (size_t i = 0; i < suffix_results.size(); ++i) {
      auto& suffix = suffix_results[i];

      if (options.return_scores) {
        result.scores[i] += suffix.scores[0];
      }

      if (options.return_attention)
        result.attention[i].insert(result.attention[i].end(),
                                   std::make_move_iterator(suffix.attention[0].begin()),
                                   std::make_move_iterator(suffix.attention[0].end()));

      result.hypotheses[i].insert(result.hypotheses[i].end(),
                                  std::make_move_iterator(suffix.hypotheses[0].begin()),
                                  std::make_move_iterator(suffix.hypotheses[0].end()));
    }

    return result;
  }

  static std::vector<size_t> map_to_output_word_ids(const layers::Decoder& decoder,
                                                    const std::vector<size_t>& ids) {
    std::vector<size_t> new_ids;
    new_ids.reserve(ids.size());
    for (const size_t id : ids) {
      if (decoder.is_in_output(id))
        new_ids.push_back(decoder.to_output_word_id(id));
    }
    return new_ids;
  }

  std::vector<DecodingResult>
  decode(layers::Decoder& decoder,
         layers::DecoderState& state,
         std::vector<std::vector<size_t>> start_tokens,
         size_t end_id,
         DecodingOptions options) {
    validate_decoding_options(options);
    const size_t batch_size = start_tokens.size();

    if (batch_size == 0)
      throw std::invalid_argument("No decoder start tokens are set");

    std::vector<DecodingResult> results;

    if (decoder.output_layer_is_updated()) {
      end_id = decoder.to_output_word_id(end_id);

      for (auto& ids : start_tokens)
        ids = map_to_output_word_ids(decoder, ids);
      for (auto& ids : options.disable_sequences)
        ids = map_to_output_word_ids(decoder, ids);

      options.disable_ids = map_to_output_word_ids(decoder, options.disable_ids);
      options.disable_ids_begin = map_to_output_word_ids(decoder, options.disable_ids_begin);
    }

    if (options.return_alternatives) {
      results.reserve(batch_size);
      for (size_t i = 0; i < batch_size; ++i) {
        layers::DecoderState batch_state = get_batch_state(state, i);
        results.emplace_back(decode_alternatives(decoder,
                                                 batch_state,
                                                 start_tokens[i],
                                                 end_id,
                                                 options));
      }

    } else {
      std::vector<size_t> start_ids;
      std::vector<std::vector<size_t>> prefix_ids;
      std::tie(start_ids, prefix_ids) = split_start_tokens(start_tokens);

      const auto search_strategy = make_search_strategy(options);
      const auto sampler = make_sampler(options);
      const auto logits_processors = make_logits_processors(options);
      results = search_strategy->search(decoder,
                                        state,
                                        *sampler,
                                        start_ids,
                                        end_id,
                                        options.start_step,
                                        options.max_length,
                                        options.min_length,
                                        options.return_scores,
                                        options.return_attention,
                                        options.num_hypotheses,
                                        options.include_eos_in_scores,
                                        options.include_eos_in_hypotheses,
                                        logits_processors,
                                        prefix_ids.empty() ? nullptr : &prefix_ids);
    }

    for (size_t b = 0; b < batch_size; ++b) {
      auto& result = results[b];

      for (size_t i = 0; i < result.hypotheses.size(); ++i) {
        // Restore original word ids.
        if (decoder.output_layer_is_updated()) {
          for (auto& id : result.hypotheses[i])
            id = decoder.to_original_word_id(id);
        }
      }
    }

    return results;
  }

}
