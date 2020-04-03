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
    StorageView input({batch_size, max_length}, int32_t(0));
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

  static std::unique_ptr<const SearchStrategy>
  make_search_strategy(const TranslationOptions& options) {
    const SearchStrategy* strategy = nullptr;

    if (options.beam_size == 1)
      strategy = new GreedySearch();
    else
      strategy = new BeamSearch(options.beam_size, options.length_penalty);

    return std::unique_ptr<const SearchStrategy>(strategy);
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
    if (other._model)
      set_model(other._model);
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
    assert_has_model();
    const size_t batch_size = source.size();
    const bool with_prefix = !target_prefix.empty();

    // Check options and inputs.
    if (options.num_hypotheses == 0)
      throw std::invalid_argument("num_hypotheses must be > 0");
    if (options.beam_size == 0)
      throw std::invalid_argument("beam_size must be > 0");
    if (options.num_hypotheses > options.beam_size && !options.return_alternatives)
      throw std::invalid_argument("The number of hypotheses can not be greater than the beam size");
    if (options.sampling_topk != 1 && options.beam_size != 1)
      throw std::invalid_argument("Random sampling should be used with beam_size = 1");
    if (options.use_vmap && _vocabulary_map->empty())
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

    // Build the final results vector.
    for (size_t i = 0, non_empty_index = 0, with_prefix_index = 0; i < batch_size; ++i) {
      if (source[i].empty())
        final_results.emplace_back(options.num_hypotheses, options.return_attention);
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

    std::vector<TranslationResult> results;
    if (options.max_batch_size == 0
        || get_batch_size(source, options.batch_type) <= options.max_batch_size)
      results = run_batch_translation(sorted_source, nullptr,  options);
    else {
      // Translate by batch of size options.max_batch_size.
      results.reserve(source.size());

      std::vector<std::vector<std::string>> partial_source;
      partial_source.reserve(source.size());
      size_t partial_batch_size = 0;

      for (auto& tokens : sorted_source) {
        const size_t batch_size_increment = get_batch_size_increment(tokens, options.batch_type);

        if (partial_batch_size > 0
            && partial_batch_size + batch_size_increment > options.max_batch_size) {
          auto partial_results = run_batch_translation(partial_source, nullptr,  options);
          results.insert(results.end(),
                         std::make_move_iterator(partial_results.begin()),
                         std::make_move_iterator(partial_results.end()));
          partial_source.clear();
          partial_batch_size = 0;
        }

        partial_source.emplace_back(std::move(tokens));
        partial_batch_size += batch_size_increment;
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

  std::vector<TranslationResult>
  Translator::run_batch_translation(const std::vector<std::vector<std::string>>& source,
                                    const std::vector<std::vector<std::string>>* target_prefix,
                                    const TranslationOptions& options) {
    PROFILE("run_batch_translation");
    auto scoped_device_setter = _model->get_scoped_device_setter();

    std::vector<std::vector<size_t>> source_ids = tokens_to_ids(source, *_source_vocabulary);
    std::vector<std::vector<size_t>> target_prefix_ids;
    if (target_prefix)
      target_prefix_ids = tokens_to_ids(*target_prefix, *_target_vocabulary);

    const Device device = _model->device();
    std::pair<StorageView, StorageView> inputs = make_inputs(source_ids, device);
    StorageView& ids = inputs.first;
    StorageView& lengths = inputs.second;

    // Encode sequence.
    StorageView encoded(device);
    (*_encoder)(ids, lengths, encoded);

    // If set, extract the subset of candidates to generate.
    std::vector<size_t> output_ids_map;
    if (options.use_vmap && !_vocabulary_map->empty()) {
      output_ids_map = _vocabulary_map->get_candidates(source);
      _decoder->set_vocabulary_mask(
        StorageView({static_cast<dim_t>(output_ids_map.size())},
                    std::vector<int32_t>(output_ids_map.begin(), output_ids_map.end()),
                    device));
    } else {
      _decoder->reset_vocabulary_mask();
    }

    // Decode.
    layers::DecoderState state = _decoder->initial_state();
    state.emplace(std::string("memory"), std::move(encoded));
    state.emplace(std::string("memory_lengths"), std::move(lengths));
    const size_t start_id = _target_vocabulary->to_id(Vocabulary::bos_token);
    const size_t end_id = _target_vocabulary->to_id(Vocabulary::eos_token);
    const size_t batch_size = source.size();
    const std::vector<size_t> start_ids(batch_size, start_id);
    std::vector<GenerationResult<size_t>> results = decode(
      *_decoder,
      state,
      *make_search_strategy(options),
      *make_sampler(options),
      start_ids,
      target_prefix ? &target_prefix_ids : nullptr,
      !output_ids_map.empty() ? &output_ids_map : nullptr,
      end_id,
      options.max_decoding_length,
      options.min_decoding_length,
      options.num_hypotheses,
      options.return_alternatives,
      options.return_scores,
      options.return_attention);

    // Convert generated ids to tokens.
    std::vector<TranslationResult> final_results;
    final_results.reserve(results.size());
    for (GenerationResult<size_t>& result : results)
      final_results.emplace_back(make_translation_result(std::move(result), *_target_vocabulary));
    return final_results;
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
    assert_has_model();
    return _model->device();
  }

  int Translator::device_index() const {
    assert_has_model();
    return _model->device_index();
  }

  ComputeType Translator::compute_type() const {
    assert_has_model();
    return _model->compute_type();
  }

  void Translator::set_model(const std::string& model_dir) {
    Device device = Device::CPU;
    int device_index = 0;
    ComputeType compute_type = ComputeType::DEFAULT;
    if (_model) {
      device = _model->device();
      device_index = _model->device_index();
      compute_type = _model->compute_type();
    }
    set_model(models::Model::load(model_dir, device, device_index, compute_type));
  }

  void Translator::set_model(const std::shared_ptr<const models::Model>& model) {
    const auto* seq2seq_model = dynamic_cast<const models::SequenceToSequenceModel*>(model.get());
    if (!seq2seq_model)
      throw std::invalid_argument("Translator expects a model of type SequenceToSequenceModel");
    _model = model;
    auto scoped_device_setter = _model->get_scoped_device_setter();
    _encoder = seq2seq_model->make_encoder();
    _decoder = seq2seq_model->make_decoder();
    _vocabulary_map = &seq2seq_model->get_vocabulary_map();
    _source_vocabulary = &seq2seq_model->get_source_vocabulary();
    _target_vocabulary = &seq2seq_model->get_target_vocabulary();
  }

  void Translator::detach_model() {
    if (!_model)
      return;
    auto scoped_device_setter = _model->get_scoped_device_setter();
    _vocabulary_map = nullptr;
    _source_vocabulary = nullptr;
    _target_vocabulary = nullptr;
    _encoder.reset();
    _decoder.reset();
    _model.reset();
  }

  void Translator::assert_has_model() const {
    if (!_model)
      throw std::runtime_error("No model is attached to this translator");
  }


  BatchType str_to_batch_type(const std::string& batch_type) {
    if (batch_type == "examples")
      return BatchType::Examples;
    else if (batch_type == "tokens")
      return BatchType::Tokens;
    throw std::invalid_argument("Invalid batch type: " + batch_type);
  }

}
