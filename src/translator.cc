#include "ctranslate2/translator.h"

#include <algorithm>
#include <numeric>

#include "ctranslate2/decoding.h"
#include "ctranslate2/ops/ops.h"
#include "ctranslate2/profiler.h"

namespace ctranslate2 {

  static std::vector<size_t>
  tokens_to_ids(const std::vector<std::string>& tokens,
                const Vocabulary& vocab) {
    std::vector<size_t> ids;
    ids.reserve(tokens.size());
    for (const auto& token : tokens)
      ids.push_back(vocab.to_id(token));
    return ids;
  }

  static std::vector<std::vector<size_t>>
  tokens_to_ids(const std::vector<std::vector<std::string>>& batch_tokens,
                const Vocabulary& vocab) {
    std::vector<std::vector<size_t>> batch_ids;
    batch_ids.reserve(batch_tokens.size());
    for (const auto& tokens : batch_tokens)
      batch_ids.emplace_back(tokens_to_ids(tokens, vocab));
    return batch_ids;
  }

  template <typename T>
  static std::vector<std::vector<T>>
  sort_from_longest_to_shortest(const std::vector<std::vector<T>>& ids,
                                std::vector<size_t>& original_to_sorted_index) {
    std::vector<size_t> sorted_to_original_index(ids.size());
    std::iota(sorted_to_original_index.begin(), sorted_to_original_index.end(), 0);
    std::sort(sorted_to_original_index.begin(), sorted_to_original_index.end(),
              [&ids](size_t i1, size_t i2) { return ids[i1].size() > ids[i2].size(); });

    original_to_sorted_index.resize(ids.size());
    std::vector<std::vector<T>> new_ids;
    new_ids.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
      size_t original_index = sorted_to_original_index[i];
      original_to_sorted_index[original_index] = i;
      new_ids.emplace_back(ids[original_index]);
    }
    return new_ids;
  }

  static std::pair<StorageView, StorageView>
  make_inputs(const std::vector<std::vector<size_t>>& ids, Device device) {
    const dim_t batch_size = ids.size();

    // Record lengths and maximum length.
    dim_t max_length = 0;
    StorageView lengths({batch_size}, DataType::INT32);
    for (dim_t i = 0; i < batch_size; ++i) {
      const dim_t length = ids[i].size();
      lengths.at<int32_t>(i) = length;
      max_length = std::max(max_length, length);
    }

    // Make 2D input.
    StorageView input({batch_size, max_length}, DataType::INT32);
    for (dim_t i = 0; i < batch_size; ++i) {
      const dim_t length = ids[i].size();
      for (dim_t t = 0; t < length; ++t)
        input.at<int32_t>({i, t}) = ids[i][t];
    }

    return std::make_pair(input.to(device), lengths.to(device));
  }

  static std::unique_ptr<const Sampler> make_sampler(const TranslationOptions& options) {
    const Sampler* sampler = nullptr;

    if (options.sampling_topk != 1)
      sampler = new RandomSampler(options.sampling_topk, options.sampling_temperature);
    else
      sampler = new BestSampler();

    return std::unique_ptr<const Sampler>(sampler);
  }


  Translator::Translator(const std::string& model_dir,
                         Device device,
                         int device_index,
                         ComputeType compute_type) {
    set_model(models::Model::load(model_dir, device, device_index, compute_type));
  }

  Translator::Translator(const std::shared_ptr<const models::Model>& model) {
    set_model(model);
  }

  Translator::Translator(const Translator& other) {
    set_model(other._model);
  }

  void Translator::make_graph() {
    auto scoped_device_setter = _model->get_scoped_device_setter();
    _encoder = _model->make_encoder();
    _decoder = _model->make_decoder();
  }

  TranslationResult
  Translator::translate(const std::vector<std::string>& tokens) {
    TranslationOptions options;
    return translate(tokens, options);
  }

  TranslationResult
  Translator::translate(const std::vector<std::string>& tokens,
                        const TranslationOptions& options) {
    std::vector<std::vector<std::string>> batch_tokens(1, tokens);
    return translate_batch(batch_tokens, options)[0];
  }

  TranslationResult
  Translator::translate_with_prefix(const std::vector<std::string>& source,
                                    const std::vector<std::string>& target_prefix,
                                    const TranslationOptions& options) {
    std::vector<std::vector<std::string>> batch_source(1, source);
    std::vector<std::vector<std::string>> batch_target_prefix(1, target_prefix);
    return translate_batch_with_prefix(batch_source, batch_target_prefix, options)[0];
  }

  std::vector<TranslationResult>
  Translator::translate_batch(const std::vector<std::vector<std::string>>& batch_tokens) {
    TranslationOptions options;
    return translate_batch(batch_tokens, options);
  }

  std::vector<TranslationResult>
  Translator::translate_batch(const std::vector<std::vector<std::string>>& batch_tokens,
                              const TranslationOptions& options) {
    std::vector<std::vector<std::string>> target_prefix;
    return translate_batch_with_prefix(batch_tokens, target_prefix, options);
  }

  std::vector<TranslationResult>
  Translator::translate_batch_with_prefix(const std::vector<std::vector<std::string>>& source,
                                          const std::vector<std::vector<std::string>>& target_prefix,
                                          const TranslationOptions& options) {
    const size_t batch_size = source.size();
    const bool with_prefix = !target_prefix.empty();

    // Check options and inputs.
    if (options.num_hypotheses > options.beam_size && !options.return_alternatives)
      throw std::invalid_argument("The number of hypotheses can not be greater than the beam size");
    if (options.sampling_topk != 1 && options.beam_size != 1)
      throw std::invalid_argument("Random sampling should be used with beam_size = 1");
    if (options.use_vmap && _model->get_vocabulary_map().empty())
      throw std::invalid_argument("use_vmap is set but the model does not include a vocabulary map");
    if (options.min_decoding_length > options.max_decoding_length)
      throw std::invalid_argument("min_decoding_length is greater than max_decoding_length");
    if (with_prefix && target_prefix.size() != batch_size)
      throw std::invalid_argument("Batch size mismatch: got "
                                  + std::to_string(batch_size) + " for source and "
                                  + std::to_string(target_prefix.size()) + " for target prefix");

    if (batch_size == 0)
      return std::vector<TranslationResult>();

    const bool no_source_is_empty = std::none_of(source.begin(),
                                                 source.end(),
                                                 [](const std::vector<std::string>& tokens) {
                                                   return tokens.empty();
                                                 });

    // Directly run translation if all source inputs are non empty and there is no target prefix.
    if (no_source_is_empty && !with_prefix)
      return run_batch_translation_sorted(source, options);

    std::vector<TranslationResult> with_prefix_results;
    std::vector<std::vector<std::string>> non_empty_source;
    with_prefix_results.reserve(batch_size);
    non_empty_source.reserve(batch_size);

    // As we don't support batch target prefix, we translate those examples separately.
    for (size_t i = 0; i < batch_size; ++i) {
      if (source[i].empty())
        continue;
      else if (with_prefix && !target_prefix[i].empty())
        with_prefix_results.emplace_back(run_translation(source[i], &target_prefix[i], options));
      else
        non_empty_source.emplace_back(source[i]);
    }

    // Run batch translation of all other non empty examples.
    std::vector<TranslationResult> results;
    if (!non_empty_source.empty())
      results = run_batch_translation_sorted(non_empty_source, options);
    std::vector<TranslationResult> final_results;
    final_results.reserve(batch_size);

    const std::vector<std::vector<std::vector<float>>> empty_attention(options.num_hypotheses);

    // Build the final results vector.
    for (size_t i = 0, non_empty_index = 0, with_prefix_index = 0; i < batch_size; ++i) {
      if (source[i].empty())
        final_results.emplace_back(std::vector<std::vector<std::string>>(options.num_hypotheses),
                                   std::vector<float>(options.num_hypotheses, static_cast<float>(0)),
                                   options.return_attention ? &empty_attention : nullptr);
      else if (with_prefix && !target_prefix[i].empty())
        final_results.emplace_back(std::move(with_prefix_results[with_prefix_index++]));
      else
        final_results.emplace_back(std::move(results[non_empty_index++]));
    }

    return final_results;
  }

  std::vector<TranslationResult>
  Translator::run_batch_translation_sorted(const std::vector<std::vector<std::string>>& source,
                                           const TranslationOptions& options) {
    // Sorting the source input has 2 benefits:
    //
    // 1. When max_batch_size is smaller that the number of inputs, we prefer translating
    //    together sentences that have a similar length for improved efficiency.
    // 2. Decoding functions remove finished translations from the batch. On CPU, arrays are
    //    updated in place so it is more efficient to remove content at the end. Shorter sentences
    //    are more likely to finish first so we sort the batch accordingly.
    std::vector<size_t> sorted_index;
    auto sorted_source = sort_from_longest_to_shortest(source, sorted_index);

    const size_t total_batch_size = source.size();
    std::vector<TranslationResult> results;

    if (options.max_batch_size == 0 || options.max_batch_size >= total_batch_size)
      results = run_batch_translation(sorted_source, nullptr,  options);
    else {
      // Translate by batch of size options.max_batch_size.
      results.reserve(total_batch_size);

      std::vector<std::vector<std::string>> partial_source;
      partial_source.reserve(options.max_batch_size);

      for (auto& tokens : sorted_source) {
        partial_source.emplace_back(std::move(tokens));

        if (partial_source.size() == options.max_batch_size) {
          auto partial_results = run_batch_translation(partial_source, nullptr,  options);
          results.insert(results.end(),
                         std::make_move_iterator(partial_results.begin()),
                         std::make_move_iterator(partial_results.end()));
          partial_source.clear();
        }
      }

      if (!partial_source.empty()) {
        auto partial_results = run_batch_translation(partial_source, nullptr,  options);
        results.insert(results.end(),
                       std::make_move_iterator(partial_results.begin()),
                       std::make_move_iterator(partial_results.end()));
      }
    }

    // Reorder results based on original batch index.
    std::vector<TranslationResult> final_results;
    final_results.reserve(results.size());
    for (auto index : sorted_index)
      final_results.emplace_back(std::move(results[index]));
    return final_results;
  }

  static void repeat_batch(StorageView& input, const dim_t repeat) {
    StorageView repeats({input.rank()}, static_cast<int32_t>(1));
    repeats.at<int32_t>(0) = repeat;
    ops::Tile()(input, repeats);
  }

  template <typename T>
  static std::vector<std::vector<T>> batch_to_hypotheses(std::vector<std::vector<T>>& array) {
    if (array.empty())
      return array;
    std::vector<std::vector<T>> new_array;
    new_array.emplace_back();
    new_array.front().reserve(array.size());
    for (auto& vector : array) {
      new_array.front().emplace_back(std::move(vector[0]));
    }
    return new_array;
  }

  std::vector<TranslationResult>
  Translator::run_batch_translation(const std::vector<std::vector<std::string>>& source,
                                    const std::vector<std::vector<std::string>>* target_prefix,
                                    const TranslationOptions& options) {
    PROFILE("run_batch_translation");

    const auto& source_vocab = _model->get_source_vocabulary();
    const auto& target_vocab = _model->get_target_vocabulary();
    const auto& vocab_map = _model->get_vocabulary_map();
    auto& encoder = *_encoder;
    auto& decoder = *_decoder;

    const dim_t batch_size = source.size();

    auto scoped_device_setter = _model->get_scoped_device_setter();
    auto device = _model->device();

    auto source_ids = tokens_to_ids(source, source_vocab);
    auto inputs = make_inputs(source_ids, device);
    StorageView& ids = inputs.first;
    StorageView& lengths = inputs.second;

    // Encode sequence.
    StorageView encoded(device);
    encoder(ids, lengths, encoded);

    // If set, extract the subset of candidates to generate.
    StorageView candidates(DataType::INT32, device);
    if (options.use_vmap && !vocab_map.empty()) {
      auto candidates_vec = vocab_map.get_candidates<int32_t>(source);
      candidates.resize({static_cast<dim_t>(candidates_vec.size())});
      candidates.copy_from(candidates_vec.data(), candidates_vec.size(), Device::CPU);
    }
    decoder.reduce_vocab(candidates);

    // Decode.
    size_t start_step = 0;
    size_t start_token = target_vocab.to_id(Vocabulary::bos_token);
    size_t end_token = target_vocab.to_id(Vocabulary::eos_token);
    StorageView sample_from({batch_size}, static_cast<int32_t>(start_token));
    std::vector<std::vector<std::vector<size_t>>> sampled_ids;
    std::vector<std::vector<float>> scores;
    std::vector<std::vector<std::vector<std::vector<float>>>> attention;
    auto* attention_ptr = options.return_attention ? &attention : nullptr;
    auto state = decoder.initial_state();

    // Forward target prefix, if set (only batch_size = 1 for now).
    std::vector<std::vector<std::vector<float>>> prefix_attention;
    if (target_prefix) {
      if (batch_size > 1)
        throw std::invalid_argument("Batched prefixed translation is not supported");

      // TODO: Forward all timesteps at once. This requires supporting the masking
      // of future steps.
      const auto& prefix = target_prefix->front();
      auto prefix_ids = tokens_to_ids(prefix, target_vocab);
      start_step = prefix.size();
      StorageView attention_step(device);
      if (options.return_attention) {
        prefix_attention.resize(1);
        prefix_attention[0].reserve(start_step);
      }

      for (size_t i = 0; i < start_step; ++i) {
        auto input = sample_from.to(device);
        input.reshape({batch_size, 1});
        decoder(i,
                input,
                encoded,
                lengths,
                state,
                /*logits=*/nullptr,
                options.return_attention ? &attention_step : nullptr);
        if (attention_step)
          prefix_attention[0].emplace_back(attention_step.to_vector<float>());
        sample_from.at<int32_t>(0) = prefix_ids[i];
      }
    }

    std::vector<std::vector<std::vector<size_t>>> expanded_ids;
    std::vector<std::vector<float>> expanded_scores;
    std::vector<std::vector<std::vector<std::vector<float>>>> expanded_attention;
    if (options.return_alternatives) {
      // In this translation mode, we first expand the next "num_hypotheses" candidate words
      // before running the full decoding on each prefix. This is to ensure that we get unique
      // alternatives at this decoding position.
      beam_search(decoder,
                  state,
                  BestSampler(),
                  sample_from,
                  candidates,
                  encoded,
                  lengths,
                  start_step,
                  end_token,
                  /*max_length=*/1,
                  /*min_length=*/1,
                  /*beam_size=*/options.num_hypotheses,
                  options.num_hypotheses,
                  /*length_penalty=*/0,
                  sampled_ids,
                  scores,
                  attention_ptr);

      start_step += 1;

      const dim_t new_batch_size = options.num_hypotheses;

      // The next input is the words we just expanded.
      sample_from.resize({new_batch_size});
      for (dim_t i = 0; i < new_batch_size; ++i) {
        sample_from.at<int32_t>(i) = sampled_ids[0][i].back();
      }

      // We are increasing the batch size from 1 to "num_hypotheses" so we need to adapt
      // some values. Note: the state was already repeated by the beam search.
      repeat_batch(encoded, new_batch_size);
      repeat_batch(lengths, new_batch_size);

      // Save expansion output as we would need to include it in the final result.
      expanded_ids = std::move(sampled_ids);
      expanded_scores = std::move(scores);
      if (attention_ptr)
        expanded_attention = std::move(*attention_ptr);
    }

    auto sampler = make_sampler(options);
    if (options.beam_size == 1)
      greedy_search(decoder,
                    state,
                    *sampler,
                    sample_from,
                    candidates,
                    encoded,
                    lengths,
                    start_step,
                    end_token,
                    options.max_decoding_length,
                    options.min_decoding_length,
                    sampled_ids,
                    scores,
                    attention_ptr);
    else
      beam_search(decoder,
                  state,
                  *sampler,
                  sample_from,
                  candidates,
                  encoded,
                  lengths,
                  start_step,
                  end_token,
                  options.max_decoding_length,
                  options.min_decoding_length,
                  options.beam_size,
                  options.return_alternatives ? 1 : options.num_hypotheses,
                  options.length_penalty,
                  sampled_ids,
                  scores,
                  attention_ptr);

    if (options.return_alternatives) {
      // We convert outputs from shape num_hypotheses x 1 to 1 x num_hypotheses.
      sampled_ids = batch_to_hypotheses(sampled_ids);
      scores = batch_to_hypotheses(scores);
      if (attention_ptr)
        *attention_ptr = batch_to_hypotheses(*attention_ptr);
    }

    // Build results.
    std::vector<TranslationResult> results;
    results.reserve(batch_size);
    for (dim_t i = 0; i < batch_size; ++i) {
      std::vector<std::vector<std::string>> hypotheses;
      size_t num_hypotheses = sampled_ids[i].size();
      hypotheses.resize(num_hypotheses);
      for (size_t h = 0; h < num_hypotheses; ++h) {
        // Finalize the hypothesis.
        const std::vector<size_t>& prediction = sampled_ids[i][h];
        std::vector<std::string>& hypothesis = hypotheses[h];
        hypothesis.reserve(prediction.size()
                           + (target_prefix ? target_prefix->at(i).size() : 0)
                           + (!expanded_ids.empty() ? 1 : 0));
        if (target_prefix)
          hypothesis.insert(hypothesis.end(),
                            target_prefix->at(i).begin(),
                            target_prefix->at(i).end());
        if (!expanded_ids.empty())
          hypothesis.push_back(target_vocab.to_token(expanded_ids[i][h][0]));
        for (const size_t id : prediction)
          hypothesis.push_back(target_vocab.to_token(id));

        // Finalize the score.
        if (!expanded_scores.empty())
          scores[i][h] += expanded_scores[i][h];

        // Finalize the attention.
        if (!prefix_attention.empty())
          attention[i][h].insert(attention[i][h].begin(),
                                 prefix_attention[i].begin(),
                                 prefix_attention[i].end());
        if (!expanded_attention.empty())
          attention[i][h].insert(attention[i][h].begin(), expanded_attention[i][h][0]);
      }
      const auto* attn = attention.empty() ? nullptr : &attention[i];
      results.emplace_back(hypotheses, scores[i], attn);
    }

    return results;
  }

  TranslationResult
  Translator::run_translation(const std::vector<std::string>& source,
                              const std::vector<std::string>* target_prefix,
                              const TranslationOptions& options) {
    if (!target_prefix)
      return run_batch_translation({source}, nullptr, options)[0];
    else {
      std::vector<std::vector<std::string>> batch_target_prefix(1, *target_prefix);
      return run_batch_translation({source}, &batch_target_prefix, options)[0];
    }
  }

  Device Translator::device() const {
    return _model->device();
  }

  int Translator::device_index() const {
    return _model->device_index();
  }

  ComputeType Translator::compute_type() const {
    return _model->compute_type();
  }

  void Translator::set_model(const std::string& model_dir) {
    set_model(models::Model::load(model_dir, device(), device_index(), compute_type()));
  }

  void Translator::set_model(const std::shared_ptr<const models::Model>& model) {
    _model = model;
    make_graph();
  }
}
