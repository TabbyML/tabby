#include "ctranslate2/models/sequence_to_sequence.h"

#include <algorithm>

namespace ctranslate2 {
  namespace models {

    static const std::string shared_vocabulary_file = "shared_vocabulary.txt";
    static const std::string source_vocabulary_file = "source_vocabulary.txt";
    static const std::string target_vocabulary_file = "target_vocabulary.txt";
    static const std::string vmap_file = "vmap.txt";
    static const std::string features_separator = "￨";

    static std::vector<std::vector<std::vector<std::string>>>
    extract_features(std::vector<std::vector<std::string>> batch, size_t num_features) {
      std::vector<std::vector<std::vector<std::string>>> features;
      features.resize(num_features);

      if (num_features == 1) {
        features[0] = std::move(batch);
        return features;
      }

      for (const auto& tokens : batch) {
        for (auto& stream : features) {
          stream.emplace_back();
          stream.back().reserve(tokens.size());
        }

        for (const auto& token : tokens) {
          auto fields = split_string(token, features_separator);
          if (fields.size() != num_features)
            throw std::invalid_argument("Expected " + std::to_string(num_features)
                                        + " input features, but token '" + token
                                        + "' has " + std::to_string(fields.size())
                                        + " features");

          for (size_t i = 0; i < fields.size(); ++i)
            features[i].back().emplace_back(std::move(fields[i]));
        }
      }

      return features;
    }


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

    const std::string* SequenceToSequenceModel::decoder_start_token() const {
      if (_user_decoder_start_tokens)
        return nullptr;
      return _with_target_bos ? &_target_vocabulary->bos_token() : &_target_vocabulary->eos_token();
    }

    const Vocabulary& SequenceToSequenceModel::get_source_vocabulary() const {
      return *_source_vocabularies[0];
    }

    const Vocabulary& SequenceToSequenceModel::get_target_vocabulary() const {
      return *_target_vocabulary;
    }

    const VocabularyMap* SequenceToSequenceModel::get_vocabulary_map() const {
      return _vocabulary_map.get();
    }

    void SequenceToSequenceModel::forward_encoder(layers::Encoder& encoder,
                                                  const std::vector<std::vector<std::vector<std::string>>>& source,
                                                  StorageView& memory,
                                                  StorageView& memory_lengths) const {
      const auto scoped_device_setter = get_scoped_device_setter();
      PROFILE("SequenceToSequenceModel::forward_encoder");

      const size_t num_input_features = source.size();
      if (_source_vocabularies.size() != source.size())
        throw std::runtime_error("The encoder expects "
                                 + std::to_string(num_input_features)
                                 + " input features, but "
                                 + std::to_string(_source_vocabularies.size())
                                 + " source vocabularies are loaded");

      std::vector<StorageView> ids;
      ids.reserve(num_input_features);

      for (size_t i = 0; i < num_input_features; ++i) {
        const auto tokens_ids = _source_vocabularies[i]->to_ids(source[i],
                                                                _with_source_bos,
                                                                _with_source_eos);
        ids.emplace_back(layers::make_sequence_inputs(tokens_ids,
                                                      device(),
                                                      preferred_size_multiple(),
                                                      i == 0 ? &memory_lengths : nullptr));
      }

      encoder(ids, memory_lengths, memory);
    }

    void SequenceToSequenceModel::forward_decoder(layers::Decoder& decoder,
                                                  layers::DecoderState& state,
                                                  const std::vector<std::vector<std::string>>& target,
                                                  StorageView& logits) const {
      const auto scoped_device_setter = get_scoped_device_setter();
      PROFILE("SequenceToSequenceModel::forward_decoder");

      const auto target_ids = _target_vocabulary->to_ids(target, decoder_start_token(), nullptr);

      StorageView lengths;
      const StorageView ids = layers::make_sequence_inputs(target_ids,
                                                           device(),
                                                           preferred_size_multiple(),
                                                           &lengths);

      decoder(ids, lengths, state, logits);
    }

    void SequenceToSequenceModel::forward(layers::Encoder& encoder,
                                          layers::Decoder& decoder,
                                          const std::vector<std::vector<std::vector<std::string>>>& source,
                                          const std::vector<std::vector<std::string>>& target,
                                          StorageView& logits) const {
      const auto scoped_device_setter = get_scoped_device_setter();
      PROFILE("SequenceToSequenceModel::forward");
      StorageView memory(encoder.output_type(), device());
      StorageView memory_lengths(DataType::INT32, device());
      forward_encoder(encoder, source, memory, memory_lengths);

      layers::DecoderState state = decoder.initial_state(/*iterative_decoding=*/false);
      state.emplace("memory", std::move(memory));
      state.emplace("memory_lengths", std::move(memory_lengths));
      forward_decoder(decoder, state, target, logits);
    }

    std::vector<ScoringResult>
    SequenceToSequenceModel::score(layers::Encoder& encoder,
                                   layers::Decoder& decoder,
                                   const std::vector<std::vector<std::string>>& source,
                                   const std::vector<std::vector<std::string>>& target,
                                   const ScoringOptions& options) const {
      const auto scoped_device_setter = get_scoped_device_setter();
      PROFILE("SequenceToSequenceModel::score");

      if (source.empty())
        return {};

      auto source_inputs = source;
      auto target_inputs = target;
      if (options.max_input_length > 0) {
        truncate_sequences(source_inputs, options.max_input_length);
        truncate_sequences(target_inputs, options.max_input_length);
      }

      const auto source_features = extract_features(std::move(source_inputs),
                                                    encoder.num_input_features());
      const auto target_ids = _target_vocabulary->to_ids(target_inputs,
                                                         decoder_start_token(),
                                                         &_target_vocabulary->eos_token());

      StorageView memory(encoder.output_type(), device());
      StorageView memory_lengths(DataType::INT32, device());
      forward_encoder(encoder, source_features, memory, memory_lengths);

      layers::DecoderState state = decoder.initial_state(/*iterative_decoding=*/false);
      state.emplace("memory", std::move(memory));
      state.emplace("memory_lengths", std::move(memory_lengths));

      return score_sequences(decoder,
                             state,
                             target_ids,
                             *_target_vocabulary,
                             preferred_size_multiple());
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

    std::vector<GenerationResult<std::string>>
    SequenceToSequenceModel::translate(layers::Encoder& encoder,
                                       layers::Decoder& decoder,
                                       const std::vector<std::vector<std::string>>& source,
                                       const std::vector<std::vector<std::string>>& target_prefix,
                                       const TranslationOptions& options) const {
      options.validate();

      const auto scoped_device_setter = get_scoped_device_setter();
      PROFILE("SequenceToSequenceModel::translate");

      const size_t original_batch_size = source.size();
      if (original_batch_size == 0)
        return {};

      const GenerationResult<std::string> empty_result(options.num_hypotheses,
                                                       options.return_attention,
                                                       options.return_scores);
      std::vector<GenerationResult<std::string>> final_results(original_batch_size, empty_result);

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
                                                    encoder.num_input_features());

      // Encode the sequence.
      StorageView memory(encoder.output_type(), device());
      StorageView memory_lengths(DataType::INT32, device());
      forward_encoder(encoder, source_features, memory, memory_lengths);

      layers::DecoderState state = decoder.initial_state();
      state.emplace("memory", std::move(memory));
      state.emplace("memory_lengths", std::move(memory_lengths));

      std::vector<size_t> include_ids;
      std::vector<size_t> exclude_ids;
      if (options.use_vmap && _vocabulary_map)
        include_ids = _vocabulary_map->get_candidates(source_features[0]);
      if (options.disable_unk)
        exclude_ids = {_target_vocabulary->unk_id()};
      const auto* output_ids_map = decoder.update_output_layer(preferred_size_multiple(),
                                                               include_ids,
                                                               exclude_ids);

      // Decode.
      if (target_prefix_inputs.empty())
        target_prefix_inputs.resize(batch_size);
      const auto start_ids = _target_vocabulary->to_ids(target_prefix_inputs,
                                                        decoder_start_token(),
                                                        nullptr);
      const size_t end_id = _target_vocabulary->eos_id();

      std::vector<GenerationResult<size_t>> results = decode(
        decoder,
        state,
        *options.make_search_strategy(),
        *options.make_sampler(),
        start_ids,
        output_ids_map,
        end_id,
        options.max_decoding_length,
        options.min_decoding_length,
        options.num_hypotheses,
        options.return_alternatives,
        options.return_scores,
        options.return_attention || options.replace_unknowns,
        options.normalize_scores,
        options.repetition_penalty);

      // Convert generated ids to tokens.
      for (size_t i = 0; i < batch_size; ++i) {
        const size_t original_index = non_empty_index[i];
        GenerationResult<size_t>& result = results[i];
        auto hypotheses = _target_vocabulary->to_tokens(result.hypotheses);

        if (result.has_attention()) {
          // Remove padding and special tokens in attention vectors.
          const size_t offset = size_t(_with_source_bos);
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
                                     _target_vocabulary->unk_token());
          }

          if (!options.return_attention)
            result.attention.clear();
        }

        final_results[original_index] = GenerationResult<std::string>(std::move(hypotheses),
                                                                      std::move(result.scores),
                                                                      std::move(result.attention));
      }

      return final_results;
    }

  }
}
