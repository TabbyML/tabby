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

    EncoderDecoderReplica::EncoderDecoderReplica(const std::shared_ptr<const SequenceToSequenceModel>& model,
                                                 std::unique_ptr<layers::Encoder> encoder,
                                                 std::unique_ptr<layers::Decoder> decoder)
      : SequenceToSequenceReplica(model)
      , _model(model)
      , _encoder(std::move(encoder))
      , _decoder(std::move(decoder))
    {
    }

    std::vector<std::vector<size_t>>
    EncoderDecoderReplica::make_source_ids(const std::vector<std::vector<std::string>>& source,
                                           size_t index) const {
      return _model->get_source_vocabulary(index).to_ids(source,
                                                         _model->with_source_bos(),
                                                         _model->with_source_eos());
    }

    std::vector<std::vector<size_t>>
    EncoderDecoderReplica::make_target_ids(const std::vector<std::vector<std::string>>& target,
                                           bool partial) const {
      const auto& target_vocabulary = _model->get_target_vocabulary();
      const std::string* suffix = partial ? nullptr : &target_vocabulary.eos_token();
      const std::string* prefix = nullptr;
      if (!_model->user_decoder_start_tokens()) {
        if (_model->with_target_bos())
          prefix = &target_vocabulary.bos_token();
        else
          prefix = &target_vocabulary.eos_token();
      }
      return target_vocabulary.to_ids(target, prefix, suffix);
    }

    void
    EncoderDecoderReplica::encode(const std::vector<std::vector<std::vector<std::string>>>& source,
                                  StorageView& memory,
                                  StorageView& memory_lengths) {
      const size_t num_input_features = source.size();
      if (_model->num_source_vocabularies() != num_input_features)
        throw std::runtime_error("The encoder expects "
                                 + std::to_string(num_input_features)
                                 + " input features, but "
                                 + std::to_string(_model->num_source_vocabularies())
                                 + " source vocabularies are loaded");

      std::vector<StorageView> ids;
      ids.reserve(num_input_features);

      for (size_t i = 0; i < num_input_features; ++i) {
        const auto tokens_ids = make_source_ids(source[i], i);
        ids.emplace_back(layers::make_sequence_inputs(tokens_ids,
                                                      _model->device(),
                                                      _model->preferred_size_multiple(),
                                                      i == 0 ? &memory_lengths : nullptr));
      }

      (*_encoder)(ids, memory_lengths, memory);
    }

    std::vector<ScoringResult>
    EncoderDecoderReplica::score(const std::vector<std::vector<std::string>>& source,
                                 const std::vector<std::vector<std::string>>& target,
                                 const ScoringOptions& options) {
      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const auto device = _model->device();
      PROFILE("EncoderDecoderReplica::score");

      if (source.empty())
        return {};

      auto source_inputs = source;
      auto target_inputs = target;
      if (options.max_input_length > 0) {
        truncate_sequences(source_inputs, options.max_input_length);
        truncate_sequences(target_inputs, options.max_input_length);
      }

      const auto source_features = extract_features(std::move(source_inputs),
                                                    _encoder->num_input_features());

      StorageView memory(_encoder->output_type(), device);
      StorageView memory_lengths(DataType::INT32, device);
      encode(source_features, memory, memory_lengths);

      layers::DecoderState state = _decoder->initial_state(/*iterative_decoding=*/false);
      state.emplace("memory", std::move(memory));
      state.emplace("memory_lengths", std::move(memory_lengths));

      return score_sequences(*_decoder,
                             state,
                             make_target_ids(target_inputs, /*partial=*/false),
                             _model->get_target_vocabulary(),
                             _model->preferred_size_multiple());
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
    EncoderDecoderReplica::translate(const std::vector<std::vector<std::string>>& source,
                                     const std::vector<std::vector<std::string>>& target_prefix,
                                     const TranslationOptions& options) {
      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const auto device = _model->device();
      PROFILE("EncoderDecoderReplica::translate");

      const size_t original_batch_size = source.size();
      if (original_batch_size == 0)
        return {};

      const TranslationResult empty_result(options.num_hypotheses,
                                           options.return_attention,
                                           options.return_scores);
      std::vector<TranslationResult> final_results(original_batch_size, empty_result);

      std::vector<size_t> non_empty_index;
      non_empty_index.reserve(original_batch_size);
      for (size_t i = 0; i < original_batch_size; ++i) {
        if (!source[i].empty())
          non_empty_index.emplace_back(i);
      }

      const size_t batch_size = non_empty_index.size();
      if (batch_size == 0)
        return final_results;

      auto source_inputs = source;
      auto target_prefix_inputs = target_prefix;
      if (batch_size != original_batch_size) {
        source_inputs = index_vector(source_inputs, non_empty_index);
        if (!target_prefix.empty())
          target_prefix_inputs = index_vector(target_prefix_inputs, non_empty_index);
      }
      if (options.max_input_length > 0) {
        truncate_sequences(source_inputs, options.max_input_length);
        truncate_sequences(target_prefix_inputs, options.max_input_length);
      }

      const auto source_features = extract_features(std::move(source_inputs),
                                                    _encoder->num_input_features());

      // Encode the sequence.
      StorageView memory(_encoder->output_type(), device);
      StorageView memory_lengths(DataType::INT32, device);
      encode(source_features, memory, memory_lengths);

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
      if (target_prefix_inputs.empty())
        target_prefix_inputs.resize(batch_size);
      const auto start_ids = make_target_ids(target_prefix_inputs, /*partial=*/true);
      const size_t end_id = target_vocabulary.eos_id();

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
                                                   start_ids,
                                                   end_id,
                                                   decoding_options,
                                                   output_ids_map);

      // Convert generated ids to tokens.
      for (size_t i = 0; i < batch_size; ++i) {
        const size_t original_index = non_empty_index[i];
        DecodingResult& result = results[i];
        auto hypotheses = target_vocabulary.to_tokens(result.hypotheses);

        if (!result.attention.empty()) {
          // Remove padding and special tokens in attention vectors.
          const size_t offset = size_t(_model->with_source_bos());
          const auto& source_original = source[original_index];
          const auto& source_input = source_features[0][i];

          for (size_t h = 0; h < result.attention.size(); ++h) {
            auto& attention = result.attention[h];

            for (auto& vector : attention) {
              vector = std::vector<float>(vector.begin() + offset,
                                          vector.begin() + offset + source_input.size());
              vector.resize(source_original.size(), 0);
            }

            if (options.replace_unknowns)
              replace_unknown_tokens(source_input,
                                     hypotheses[h],
                                     attention,
                                     target_vocabulary.unk_token());
          }

          if (!options.return_attention)
            result.attention.clear();
        }

        final_results[original_index] = TranslationResult(std::move(hypotheses),
                                                          std::move(result.scores),
                                                          std::move(result.attention));
      }

      return final_results;
    }

  }
}
