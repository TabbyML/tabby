#include "ctranslate2/models/sequence_to_sequence.h"

#include <algorithm>

#include "ctranslate2/decoding.h"

namespace ctranslate2 {
  namespace models {

    static const std::string shared_vocabulary_file = "shared_vocabulary.txt";
    static const std::string source_vocabulary_file = "source_vocabulary.txt";
    static const std::string target_vocabulary_file = "target_vocabulary.txt";
    static const std::string vmap_file = "vmap.txt";


    void SequenceToSequenceModel::load_vocabularies(ModelReader& model_reader) {
      {
        VocabularyInfo vocab_info;
        vocab_info.unk_token = get_attribute_with_default<std::string>("unk_token", "<unk>");
        vocab_info.bos_token = get_attribute_with_default<std::string>("bos_token", "<s>");
        vocab_info.eos_token = get_attribute_with_default<std::string>("eos_token", "</s>");

        auto shared_vocabulary = model_reader.get_file(shared_vocabulary_file);
        if (shared_vocabulary) {
          _target_vocabulary = std::make_shared<Vocabulary>(*shared_vocabulary, vocab_info);
          _source_vocabularies.emplace_back(_target_vocabulary);
        } else {

          {
            auto source_vocabulary = model_reader.get_file(source_vocabulary_file);
            if (source_vocabulary)
              _source_vocabularies.emplace_back(std::make_shared<Vocabulary>(*source_vocabulary,
                                                                             vocab_info));
            else {
              for (size_t i = 1;; i++) {
                const std::string filename = "source_" + std::to_string(i) + "_vocabulary.txt";
                const auto vocabulary_file = model_reader.get_file(filename);
                if (!vocabulary_file)
                  break;
                _source_vocabularies.emplace_back(std::make_shared<Vocabulary>(*vocabulary_file,
                                                                               vocab_info));
              }
            }

            // If no source vocabularies were loaded, raise an error for the first filename.
            if (_source_vocabularies.empty())
              model_reader.get_required_file(source_vocabulary_file);
          }

          {
            auto target_vocabulary = model_reader.get_required_file(target_vocabulary_file);
            _target_vocabulary = std::make_shared<Vocabulary>(*target_vocabulary, vocab_info);
          }
        }
      }

      {
        auto vmap = model_reader.get_file(vmap_file);
        if (vmap) {
          _vocabulary_map = std::make_unique<VocabularyMap>(*vmap, get_target_vocabulary());
        }
      }
    }

    void SequenceToSequenceModel::initialize(ModelReader& model_reader) {
      Model::initialize(model_reader);
      load_vocabularies(model_reader);
      _with_source_bos = get_flag_with_default("with_source_bos", false);
      _with_source_eos = get_flag_with_default("with_source_eos", false);
      _with_target_bos = get_flag_with_default("with_target_bos", true);
      _user_decoder_start_tokens = get_flag_with_default("user_decoder_start_tokens", false);
    }

    size_t SequenceToSequenceModel::num_source_vocabularies() const {
      return _source_vocabularies.size();
    }

    const Vocabulary& SequenceToSequenceModel::get_source_vocabulary(size_t index) const {
      return *_source_vocabularies.at(index);
    }

    const Vocabulary& SequenceToSequenceModel::get_target_vocabulary() const {
      return *_target_vocabulary;
    }

    const VocabularyMap* SequenceToSequenceModel::get_vocabulary_map() const {
      return _vocabulary_map.get();
    }


    std::vector<ScoringResult>
    SequenceToSequenceReplica::score(const std::vector<std::vector<std::string>>& source,
                                     const std::vector<std::vector<std::string>>& target,
                                     const ScoringOptions& options) {
      return get_batch_results_helper<ScoringResult>(
        source.size(),
        [this, &source, &target, &options](size_t i, ScoringResult& result) {
          return skip_scoring(source[i], target[i], options, result);
        },
        [this, &source, &target, &options](const std::vector<size_t>& index_to_run) {
          return run_scoring(index_vector(source, index_to_run),
                             index_vector(target, index_to_run),
                             options);
        });
    }

    std::vector<TranslationResult>
    SequenceToSequenceReplica::translate(const std::vector<std::vector<std::string>>& source,
                                         const std::vector<std::vector<std::string>>& target_prefix,
                                         const TranslationOptions& options) {
      auto target = target_prefix;
      if (target.empty())
        target.resize(source.size());

      return get_batch_results_helper<TranslationResult>(
        source.size(),
        [this, &source, &target, &options](size_t i, TranslationResult& result) {
          return skip_translation(source[i], target[i], options, result);
        },
        [this, &source, &target, &options](const std::vector<size_t>& index_to_run) {
          return run_translation(index_vector(source, index_to_run),
                                 index_vector(target, index_to_run),
                                 options);
        });
    }


    EncoderDecoderReplica::EncoderDecoderReplica(const std::shared_ptr<const SequenceToSequenceModel>& model,
                                                 std::unique_ptr<layers::Encoder> encoder,
                                                 std::unique_ptr<layers::Decoder> decoder)
      : SequenceToSequenceReplica(model)
      , _model(model)
      , _encoder(std::move(encoder))
      , _decoder(std::move(decoder))
    {
    }

    std::vector<std::vector<std::vector<size_t>>>
    EncoderDecoderReplica::make_source_ids(const std::vector<std::vector<std::vector<std::string>>>& source_features,
                                           size_t max_length) const {
      const size_t num_input_features = source_features.size();
      if (_model->num_source_vocabularies() != num_input_features)
        throw std::runtime_error("The encoder expects "
                                 + std::to_string(num_input_features)
                                 + " input features, but "
                                 + std::to_string(_model->num_source_vocabularies())
                                 + " source vocabularies are loaded");

      std::vector<std::vector<std::vector<size_t>>> ids;
      ids.reserve(num_input_features);

      for (size_t i = 0; i < num_input_features; ++i) {
        const auto& vocabulary = _model->get_source_vocabulary(i);
        ids.emplace_back(vocabulary.to_ids(source_features[i],
                                           max_length,
                                           _model->with_source_bos(),
                                           _model->with_source_eos()));
      }

      return ids;
    }

    std::vector<std::vector<size_t>>
    EncoderDecoderReplica::make_target_ids(const std::vector<std::vector<std::string>>& target,
                                           size_t max_length,
                                           bool is_prefix) const {
      const auto& target_vocabulary = _model->get_target_vocabulary();
      const std::string* suffix = &target_vocabulary.eos_token();
      const std::string* prefix = nullptr;
      if (!_model->user_decoder_start_tokens()) {
        if (_model->with_target_bos())
          prefix = &target_vocabulary.bos_token();
        else
          prefix = &target_vocabulary.eos_token();
      }

      if (is_prefix) {
        suffix = nullptr;
        max_length = 0;
      } else if (max_length > 0) {
        // The method returns the full target "<s> a b c </s>" but the decoder input is "<s> a b c".
        // So 1 additional token is allowed in the full target sequence.
        max_length += 1;
      }

      return target_vocabulary.to_ids(target, max_length, prefix, suffix);
    }

    size_t EncoderDecoderReplica::get_source_length(const std::vector<std::string>& source,
                                                    bool include_special_tokens) const {
      size_t length = source.size();

      if (include_special_tokens) {
        if (_model->with_source_bos())
          ++length;
        if (_model->with_source_eos())
          ++length;

      } else {
        const auto& vocabulary = _model->get_source_vocabulary(0);
        if (source.size() == 1) {
          if (vocabulary.bos_token() == source[0] || vocabulary.eos_token() == source[0])
            --length;
        } else if (source.size() >= 2) {
          if (vocabulary.bos_token() == source[0])
            --length;
          if (vocabulary.eos_token() == source[source.size() - 1])
            --length;
        }
      }

      return length;
    }

    void
    EncoderDecoderReplica::encode(const std::vector<std::vector<std::vector<size_t>>>& features_ids,
                                  StorageView& memory,
                                  StorageView& memory_lengths) {
      const size_t num_input_features = features_ids.size();
      std::vector<StorageView> ids;
      ids.reserve(num_input_features);

      for (size_t i = 0; i < num_input_features; ++i) {
        const auto& tokens_ids = features_ids[i];
        ids.emplace_back(layers::make_sequence_inputs(tokens_ids,
                                                      _model->device(),
                                                      _model->preferred_size_multiple(),
                                                      i == 0 ? &memory_lengths : nullptr));
      }

      (*_encoder)(ids, memory_lengths, memory);
    }

    std::vector<ScoringResult>
    EncoderDecoderReplica::run_scoring(const std::vector<std::vector<std::string>>& source,
                                       const std::vector<std::vector<std::string>>& target,
                                       const ScoringOptions& options) {
      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const auto device = _model->device();
      PROFILE("EncoderDecoderReplica::run_scoring");

      const auto source_features = extract_features(source, _encoder->num_input_features());
      const auto source_ids = make_source_ids(source_features, options.max_input_length);
      const auto target_ids = make_target_ids(target, options.max_input_length);

      StorageView memory(_encoder->output_type(), device);
      StorageView memory_lengths(DataType::INT32, device);
      encode(source_ids, memory, memory_lengths);

      layers::DecoderState state = _decoder->initial_state(/*iterative_decoding=*/false);
      state.emplace("memory", std::move(memory));
      state.emplace("memory_lengths", std::move(memory_lengths));

      return score_sequences(*_decoder,
                             state,
                             target_ids,
                             _model->get_target_vocabulary(),
                             _model->preferred_size_multiple());
    }

    bool EncoderDecoderReplica::skip_scoring(const std::vector<std::string>& source,
                                             const std::vector<std::string>& target,
                                             const ScoringOptions& options,
                                             ScoringResult& result) {
      if (_model->user_decoder_start_tokens() && target.empty()) {
        return true;
      }

      // If the source is empty even with special tokens, we can't run the model on this input
      // so we set a score of 0 for target tokens that would be scored by the model.
      if (get_source_length(source, /*include_special_tokens=*/true) == 0) {
        const auto& vocabulary = _model->get_target_vocabulary();
        const auto target_ids = make_target_ids({target}, options.max_input_length)[0];
        result.tokens.reserve(target_ids.size() - 1);
        result.tokens_score.reserve(target_ids.size() - 1);
        for (size_t i = 1; i < target_ids.size(); ++i) {
          result.tokens.emplace_back(vocabulary.to_token(target_ids[i]));
          result.tokens_score.emplace_back(0);
        }
        return true;
      }

      return false;
    }

    static void replace_unknown_tokens(const std::vector<std::string>& source,
                                       std::vector<std::string>& hypotheses,
                                       const std::vector<std::vector<float>>& attention,
                                       const std::string& unk_token) {
      for (size_t t = 0; t < hypotheses.size(); ++t) {
        if (hypotheses[t] == unk_token) {
          const std::vector<float>& attention_values = attention[t];
          const size_t pos = std::distance(attention_values.begin(),
                                           std::max_element(attention_values.begin(),
                                                            attention_values.end()));

          hypotheses[t] = source[pos];
        }
      }
    }

    std::vector<TranslationResult>
    EncoderDecoderReplica::run_translation(const std::vector<std::vector<std::string>>& source,
                                           const std::vector<std::vector<std::string>>& target_prefix,
                                           const TranslationOptions& options) {
      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const auto device = _model->device();
      PROFILE("EncoderDecoderReplica::run_translation");

      const size_t batch_size = source.size();

      const auto source_features = extract_features(source, _encoder->num_input_features());
      const auto source_ids = make_source_ids(source_features, options.max_input_length);
      const auto target_ids = make_target_ids(target_prefix,
                                              options.max_input_length,
                                              /*is_prefix=*/true);

      // Encode the sequence.
      StorageView memory(_encoder->output_type(), device);
      StorageView memory_lengths(DataType::INT32, device);
      encode(source_ids, memory, memory_lengths);

      layers::DecoderState state = _decoder->initial_state();
      state.emplace("memory", std::move(memory));
      state.emplace("memory_lengths", std::move(memory_lengths));

      const auto& target_vocabulary = _model->get_target_vocabulary();
      std::vector<size_t> include_ids;
      std::vector<size_t> exclude_ids;
      if (options.use_vmap && _model->get_vocabulary_map())
        include_ids = _model->get_vocabulary_map()->get_candidates(source_features[0]);
      if (options.disable_unk)
        exclude_ids = {target_vocabulary.unk_id()};
      const auto* output_ids_map = _decoder->update_output_layer(_model->preferred_size_multiple(),
                                                                 include_ids,
                                                                 exclude_ids);

      // Decode.
      DecodingOptions decoding_options;
      decoding_options.beam_size = options.beam_size;
      decoding_options.length_penalty = options.length_penalty;
      decoding_options.coverage_penalty = options.coverage_penalty;
      decoding_options.repetition_penalty = options.repetition_penalty;
      decoding_options.prefix_bias_beta = options.prefix_bias_beta;
      decoding_options.allow_early_exit = options.allow_early_exit;
      decoding_options.max_length = options.max_decoding_length;
      decoding_options.min_length = options.min_decoding_length;
      decoding_options.sampling_topk = options.sampling_topk;
      decoding_options.sampling_temperature = options.sampling_temperature;
      decoding_options.num_hypotheses = options.num_hypotheses;
      decoding_options.normalize_scores = options.normalize_scores;
      decoding_options.return_scores = options.return_scores;
      decoding_options.return_attention = options.return_attention || options.replace_unknowns;
      decoding_options.return_alternatives = options.return_alternatives;

      std::vector<DecodingResult> results = decode(*_decoder,
                                                   state,
                                                   target_ids,
                                                   target_vocabulary.eos_id(),
                                                   decoding_options,
                                                   output_ids_map);

      // Convert generated ids to tokens.
      std::vector<TranslationResult> final_results;
      final_results.reserve(batch_size);

      for (size_t i = 0; i < batch_size; ++i) {
        DecodingResult& result = results[i];
        auto hypotheses = target_vocabulary.to_tokens(result.hypotheses);

        if (!result.attention.empty()) {
          const auto& source_original = source_features[0][i];
          const auto& source_input = source_ids[0][i];

          for (size_t h = 0; h < result.attention.size(); ++h) {
            auto& attention = result.attention[h];

            for (auto& vector : attention) {
              // Remove attenton positions for padding and implicit special tokens.
              vector.resize(source_input.size());
              if (_model->with_source_bos())
                vector.erase(vector.begin());
              if (_model->with_source_eos())
                vector.pop_back();

              // Resize to the original input size.
              vector.resize(source_original.size(), 0);
            }

            if (options.replace_unknowns)
              replace_unknown_tokens(source_original,
                                     hypotheses[h],
                                     attention,
                                     target_vocabulary.unk_token());
          }

          if (!options.return_attention)
            result.attention.clear();
        }

        final_results.emplace_back(std::move(hypotheses),
                                   std::move(result.scores),
                                   std::move(result.attention));
      }

      return final_results;
    }

    bool EncoderDecoderReplica::skip_translation(const std::vector<std::string>& source,
                                                 const std::vector<std::string>&,
                                                 const TranslationOptions& options,
                                                 TranslationResult& result) {
      // If the source content is empty, we assume the translation is empty.
      if (get_source_length(source, /*include_special_tokens=*/false) == 0) {
        result = TranslationResult(options.num_hypotheses,
                                   options.return_attention,
                                   options.return_scores);
        return true;
      }
      return false;
    }

  }
}
