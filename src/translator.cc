#include "ctranslate2/translator.h"

#include <algorithm>
#include <numeric>

#include "ctranslate2/decoding.h"
#include "ctranslate2/ops/ops.h"
#include "ctranslate2/profiler.h"

namespace ctranslate2 {

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
      strategy = new BeamSearch(options.beam_size, options.length_penalty, options.coverage_penalty);

    return std::unique_ptr<const SearchStrategy>(strategy);
  }


  void TranslationOptions::validate() const {
    if (num_hypotheses == 0)
      throw std::invalid_argument("num_hypotheses must be > 0");
    if (beam_size == 0)
      throw std::invalid_argument("beam_size must be > 0");
    if (num_hypotheses > beam_size && !return_alternatives)
      throw std::invalid_argument("The number of hypotheses can not be greater than the beam size");
    if (sampling_topk != 1 && beam_size != 1)
      throw std::invalid_argument("Random sampling should be used with beam_size = 1");
    if (min_decoding_length > max_decoding_length)
      throw std::invalid_argument("min_decoding_length is greater than max_decoding_length");
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
    if (!options.validated)
      options.validate();
    if (!options.rebatch_input)
      return run_batch_translation(source, target_prefix, options);

    const TranslationResult empty_result(options.num_hypotheses, options.return_attention);
    std::vector<TranslationResult> results(source.size(), empty_result);

    for (const auto& batch : rebatch_input(source, target_prefix, options)) {
      auto batch_results = run_batch_translation(batch.source, batch.target, options);
      for (size_t i = 0; i < batch_results.size(); ++i)
        results[batch.example_index[i]] = std::move(batch_results[i]);
    }

    return results;
  }

  static void
  replace_unknowns(const std::vector<std::string>& source,
                   std::vector<std::vector<std::string>>& hypotheses,
                   const std::vector<std::vector<std::vector<float>>>& attention) {
      for (size_t h = 0; h < hypotheses.size(); ++h) {
        for (size_t t = 0; t < hypotheses[h].size(); ++t) {
          if (hypotheses[h][t] == Vocabulary::unk_token) {
            const std::vector<float>& attention_values = attention[h][t];
            const size_t pos = std::distance(attention_values.begin(),
                                             std::max_element(attention_values.begin(), attention_values.end()));

            hypotheses[h][t] = source[pos];
          }
        }
      }
    }

  std::vector<TranslationResult>
  Translator::run_batch_translation(const std::vector<std::vector<std::string>>& source,
                                    const std::vector<std::vector<std::string>>& target_prefix,
                                    const TranslationOptions& options) {
    PROFILE("run_batch_translation");
    assert_has_model();
    auto scoped_device_setter = _model->get_scoped_device_setter();

    const auto& source_vocabulary = _seq2seq_model->get_source_vocabulary();
    const auto& target_vocabulary = _seq2seq_model->get_target_vocabulary();
    const auto source_ids = source_vocabulary.to_ids(source);
    const auto target_prefix_ids = target_vocabulary.to_ids(target_prefix);

    const Device device = _model->device();
    const dim_t preferred_size_multiple = get_preferred_size_multiple(
      _model->effective_compute_type(),
      device,
      _model->device_index());
    std::pair<StorageView, StorageView> inputs = layers::make_sequence_inputs(
      source_ids,
      device,
      preferred_size_multiple);
    StorageView& ids = inputs.first;
    StorageView& lengths = inputs.second;

    // Encode sequence.
    StorageView encoded(_encoder->output_type(), device);
    (*_encoder)(ids, lengths, encoded);

    // If set, extract the subset of candidates to generate.
    const auto* vocabulary_map = _seq2seq_model->get_vocabulary_map();
    std::vector<size_t> output_ids_map;
    if (options.use_vmap && vocabulary_map) {
      output_ids_map = vocabulary_map->get_candidates(source);
    } else if (target_vocabulary.size() % preferred_size_multiple != 0) {
      output_ids_map.resize(target_vocabulary.size());
      std::iota(output_ids_map.begin(), output_ids_map.end(), size_t(0));
    }

    if (!output_ids_map.empty()) {
      // Pad vocabulary size to the preferred size multiple.
      while (output_ids_map.size() % preferred_size_multiple != 0)
        output_ids_map.push_back(0);

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
    const size_t start_id = target_vocabulary.to_id(Vocabulary::bos_token);
    const size_t end_id = target_vocabulary.to_id(Vocabulary::eos_token);
    const size_t batch_size = source.size();
    const std::vector<size_t> start_ids(batch_size, start_id);
    std::vector<GenerationResult<size_t>> results = decode(
      *_decoder,
      state,
      *make_search_strategy(options),
      *make_sampler(options),
      start_ids,
      !target_prefix_ids.empty() ? &target_prefix_ids : nullptr,
      !output_ids_map.empty() ? &output_ids_map : nullptr,
      end_id,
      options.max_decoding_length,
      options.min_decoding_length,
      options.num_hypotheses,
      options.return_alternatives,
      options.return_scores,
      options.return_attention || options.replace_unknowns);

    // Convert generated ids to tokens.
    std::vector<TranslationResult> final_results;
    final_results.reserve(results.size());
    for (size_t i = 0; i < batch_size; ++i) {
      GenerationResult<size_t>& result = results[i];
      std::vector<std::vector<std::string>> hypotheses = target_vocabulary.to_tokens(result.hypotheses());
  
      if (result.has_attention() && options.replace_unknowns) {
        const auto& attention_values = results[i].attention();
        replace_unknowns(source[i], hypotheses, attention_values);

        if (!options.return_attention) {
          std::vector<std::vector<std::vector<float>>> empty_attention;
          result.set_attention(empty_attention);
        }
      }

      // Remove padding in attention vectors.
      if (result.has_attention()) {
        const size_t source_length = source[i].size();
        auto all_attention = result.attention();
        for (auto& attention : all_attention) {
          for (auto& vector : attention) {
            vector.resize(source_length);
          }
        }
        result.set_attention(std::move(all_attention));
      }


      final_results.emplace_back(hypotheses,
                                 result.scores(),
                                 result.attention());
    }
    return final_results;
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
    models::ModelFileReader model_reader(model_dir);
    set_model(model_reader);
  }

  void Translator::set_model(models::ModelReader& model_reader) {
    Device device = Device::CPU;
    int device_index = 0;
    ComputeType compute_type = ComputeType::DEFAULT;
    if (_model) {
      device = _model->device();
      device_index = _model->device_index();
      compute_type = _model->compute_type();
    }
    set_model(models::Model::load(model_reader, device, device_index, compute_type));
  }

  void Translator::set_model(const std::shared_ptr<const models::Model>& model) {
    const auto* seq2seq_model = dynamic_cast<const models::SequenceToSequenceModel*>(model.get());
    if (!seq2seq_model)
      throw std::invalid_argument("Translator expects a model of type SequenceToSequenceModel");
    _model = model;
    _seq2seq_model = seq2seq_model;
    auto scoped_device_setter = _model->get_scoped_device_setter();
    _encoder = seq2seq_model->make_encoder();
    _decoder = seq2seq_model->make_decoder();
  }

  void Translator::detach_model() {
    if (!_model)
      return;
    auto scoped_device_setter = _model->get_scoped_device_setter();
    _encoder.reset();
    _decoder.reset();
    _model.reset();
    _seq2seq_model = nullptr;
  }

  void Translator::assert_has_model() const {
    if (!_model)
      throw std::runtime_error("No model is attached to this translator");
  }

  std::vector<Batch>
  rebatch_input(const std::vector<std::vector<std::string>>& source,
                const std::vector<std::vector<std::string>>& target_prefix,
                const TranslationOptions& options) {
    size_t max_batch_size = options.max_batch_size;
    BatchType batch_type = options.batch_type;
    if (options.return_alternatives) {
      max_batch_size = 1;  // Disable batching in return_alternatives mode.
      batch_type = BatchType::Examples;
    }
    return rebatch_input(source, target_prefix, max_batch_size, batch_type);
  }

  std::vector<Batch>
  rebatch_input(const std::vector<std::vector<std::string>>& source,
                const std::vector<std::vector<std::string>>& target,
                size_t max_batch_size,
                BatchType batch_type) {
    if (!target.empty() && target.size() != source.size())
      throw std::invalid_argument("Batch size mismatch: got "
                                  + std::to_string(source.size()) + " for source and "
                                  + std::to_string(target.size()) + " for target");

    const size_t global_batch_size = source.size();
    if (max_batch_size == 0) {
      max_batch_size = global_batch_size;
      batch_type = BatchType::Examples;
    }

    // Sorting the source inputs from the longest to the shortest has 2 benefits:
    //
    // 1. When max_batch_size is smaller that the number of inputs, we prefer translating
    //    together sentences that have a similar length for improved efficiency.
    // 2. Decoding functions remove finished translations from the batch. On CPU, arrays are
    //    updated in place so it is more efficient to remove content at the end. Shorter sentences
    //    are more likely to finish first so we sort the batch accordingly.
    std::vector<size_t> example_index(global_batch_size);
    std::iota(example_index.begin(), example_index.end(), 0);
    std::sort(example_index.begin(), example_index.end(),
              [&source](size_t i1, size_t i2) {
                return source[i1].size() > source[i2].size();
              });

    // Ignore empty examples.
    // As example_index is sorted from longest to shortest, we simply pop empty examples
    // from the back.
    while (!example_index.empty() && source[example_index.back()].empty())
      example_index.pop_back();

    std::vector<Batch> batches;
    if (example_index.empty())
      return batches;
    batches.reserve(example_index.size());

    ParallelBatchReader batch_reader;
    batch_reader.add(new VectorReader(index_vector(source, example_index)));
    if (!target.empty())
      batch_reader.add(new VectorReader(index_vector(target, example_index)));

    for (size_t offset = 0;;) {
      auto batch_tokens = batch_reader.get_next(max_batch_size, batch_type);
      if (batch_tokens[0].empty())
        break;

      Batch batch;
      batch.source = std::move(batch_tokens[0]);
      if (batch_tokens.size() > 1)
        batch.target = std::move(batch_tokens[1]);

      const size_t batch_size = batch.source.size();
      batch.example_index.insert(batch.example_index.begin(),
                                 example_index.begin() + offset,
                                 example_index.begin() + offset + batch_size);
      offset += batch_size;

      batches.emplace_back(std::move(batch));
    }

    return batches;
  }

}
