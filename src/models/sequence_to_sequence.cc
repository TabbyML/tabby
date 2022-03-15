#include "ctranslate2/models/sequence_to_sequence.h"

#include <algorithm>
#include <numeric>

namespace ctranslate2 {
  namespace models {

    static const std::string shared_vocabulary_file = "shared_vocabulary.txt";
    static const std::string source_vocabulary_file = "source_vocabulary.txt";
    static const std::string target_vocabulary_file = "target_vocabulary.txt";
    static const std::string vmap_file = "vmap.txt";
    static const std::string features_separator = "￨";

    template <typename T>
    static std::vector<std::vector<T>>
    truncate_inputs(const std::vector<std::vector<T>>& inputs, size_t max_length) {
      std::vector<std::vector<T>> truncated_inputs;
      truncated_inputs.reserve(inputs.size());
      for (const auto& input : inputs)
        truncated_inputs.emplace_back(input.begin(),
                                      input.begin() + std::min(input.size(), max_length));
      return truncated_inputs;
    }

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


    SequenceToSequenceModel::SequenceToSequenceModel(ModelReader& model_reader, size_t spec_revision)
      : Model(model_reader, spec_revision) {
      {
        auto shared_vocabulary = model_reader.get_file(shared_vocabulary_file);
        if (shared_vocabulary) {
          _target_vocabulary = std::make_shared<Vocabulary>(*shared_vocabulary);
          _source_vocabularies.emplace_back(_target_vocabulary);
        } else {

          {
            auto source_vocabulary = model_reader.get_file(source_vocabulary_file);
            if (source_vocabulary)
              _source_vocabularies.emplace_back(std::make_shared<Vocabulary>(*source_vocabulary));
            else {
              for (size_t i = 1;; i++) {
                const std::string filename = "source_" + std::to_string(i) + "_vocabulary.txt";
                const auto vocabulary_file = model_reader.get_file(filename);
                if (!vocabulary_file)
                  break;
                _source_vocabularies.emplace_back(std::make_shared<Vocabulary>(*vocabulary_file));
              }
            }

            // If no source vocabularies were loaded, raise an error for the first filename.
            if (_source_vocabularies.empty())
              model_reader.get_required_file(source_vocabulary_file);
          }

          {
            auto target_vocabulary = model_reader.get_required_file(target_vocabulary_file);
            _target_vocabulary = std::make_shared<Vocabulary>(*target_vocabulary);
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

    void SequenceToSequenceModel::finalize() {
      Model::finalize();
      _with_source_bos = get_flag_with_default("with_source_bos", false);
      _with_source_eos = get_flag_with_default("with_source_eos", false);
      _with_target_bos = get_flag_with_default("with_target_bos", true);
      _user_decoder_start_tokens = get_flag_with_default("user_decoder_start_tokens", false);
    }

    const Vocabulary& SequenceToSequenceModel::get_source_vocabulary() const {
      return *_source_vocabularies[0];
    }

    const Vocabulary& SequenceToSequenceModel::get_target_vocabulary() const {
      return *_target_vocabulary;
    }

    const VocabularyMap* SequenceToSequenceModel::get_vocabulary_map() const {
      return _vocabulary_map && !_vocabulary_map->empty() ? _vocabulary_map.get() : nullptr;
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
      const auto target_ids = _target_vocabulary->to_ids(target,
                                                         /*add_bos=*/true,
                                                         /*add_eos=*/false);

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
                                   const size_t max_input_length) const {
      const auto scoped_device_setter = get_scoped_device_setter();
      PROFILE("SequenceToSequenceModel::score");

      auto source_inputs = source;
      auto target_inputs = target;
      if (max_input_length > 0) {
        source_inputs = truncate_inputs(source_inputs, max_input_length);
        target_inputs = truncate_inputs(target_inputs, max_input_length);
      }

      const auto source_features = extract_features(std::move(source_inputs),
                                                    encoder.num_input_features());

      StorageView logits(decoder.output_type(), device());
      forward(encoder, decoder, source_features, target_inputs, logits);
      StorageView log_probs = std::move(logits);
      ops::LogSoftMax()(log_probs);

      const auto target_ids_out = _target_vocabulary->to_ids(target_inputs,
                                                             /*add_bos=*/false,
                                                             /*add_eos=*/true);

      const StorageView gather_ids = layers::make_sequence_inputs(target_ids_out,
                                                                  device(),
                                                                  preferred_size_multiple());

      StorageView scores(log_probs.dtype(), device());
      ops::Gather(/*axis=*/-1, /*batch_dims=*/2)(log_probs, gather_ids, scores);

      if (scores.device() != Device::CPU)
        scores = scores.to(Device::CPU);
      if (scores.dtype() != DataType::FLOAT)
        scores = scores.to_float();

      const dim_t batch_size = scores.dim(0);
      std::vector<ScoringResult> results(batch_size);
      for (dim_t b = 0; b < batch_size; ++b) {
        const dim_t output_length = target_ids_out[b].size();
        auto& result = results[b];
        result.tokens.reserve(output_length);
        result.tokens_score.reserve(output_length);
        for (dim_t t = 0; t < output_length; ++t) {
          result.tokens.emplace_back(_target_vocabulary->to_token(target_ids_out[b][t]));
          result.tokens_score.emplace_back(scores.at<float>({b, t}));
        }
      }

      return results;
    }

    static void replace_unknown_tokens(const std::vector<std::string>& source,
                                       std::vector<std::string>& hypotheses,
                                       const std::vector<std::vector<float>>& attention) {
      for (size_t t = 0; t < hypotheses.size(); ++t) {
        if (hypotheses[t] == Vocabulary::unk_token) {
          const std::vector<float>& attention_values = attention[t];
          const size_t pos = std::distance(attention_values.begin(),
                                           std::max_element(attention_values.begin(),
                                                            attention_values.end()));

          hypotheses[t] = source[pos];
        }
      }
    }

    static inline void raise_prefix_is_required() {
      throw std::invalid_argument("This model requires a target prefix with at least "
                                  "one token corresponding to the decoder start token");
    }

    static std::vector<size_t>
    get_start_ids_from_prefix(std::vector<std::vector<size_t>>& target_prefix) {
      if (target_prefix.empty())
        raise_prefix_is_required();

      std::vector<size_t> start_ids;
      start_ids.reserve(target_prefix.size());
      bool prefix_has_only_start_token = true;

      for (auto& prefix : target_prefix) {
        if (prefix.empty())
          raise_prefix_is_required();

        start_ids.emplace_back(prefix.front());
        prefix.erase(prefix.begin());

        if (!prefix.empty())
          prefix_has_only_start_token = false;
      }

      if (prefix_has_only_start_token)
        target_prefix.clear();

      return start_ids;
    }

    std::vector<GenerationResult<std::string>>
    SequenceToSequenceModel::sample(layers::Encoder& encoder,
                                    layers::Decoder& decoder,
                                    const std::vector<std::vector<std::string>>& source,
                                    const std::vector<std::vector<std::string>>& target_prefix,
                                    const SearchStrategy& search_strategy,
                                    const Sampler& sampler,
                                    const bool use_vmap,
                                    const size_t max_input_length,
                                    const size_t max_output_length,
                                    const size_t min_output_length,
                                    const size_t num_hypotheses,
                                    const bool return_alternatives,
                                    const bool return_scores,
                                    const bool return_attention,
                                    const bool replace_unknowns,
                                    const bool normalize_scores,
                                    const float repetition_penalty,
                                    bool disable_unk) const {
      const auto scoped_device_setter = get_scoped_device_setter();
      PROFILE("SequenceToSequenceModel::sample");

      auto source_inputs = source;
      auto target_prefix_inputs = target_prefix;
      if (max_input_length > 0) {
        source_inputs = truncate_inputs(source_inputs, max_input_length);
        target_prefix_inputs = truncate_inputs(target_prefix_inputs, max_input_length);
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

      std::vector<size_t> output_ids_map;
      if (use_vmap && _vocabulary_map) {
        output_ids_map = _vocabulary_map->get_candidates(source_features[0]);
      } else if (_target_vocabulary->size() % preferred_size_multiple() != 0) {
        output_ids_map.resize(_target_vocabulary->size());
        std::iota(output_ids_map.begin(), output_ids_map.end(), size_t(0));
      }

      // If UNK generation is disabled, we can directly remove the token from the
      // reduced vocabulary instead of handling that during decoding.
      const size_t unk_id = _target_vocabulary->to_id(Vocabulary::unk_token);
      if (disable_unk && !output_ids_map.empty()) {
        auto it = std::lower_bound(output_ids_map.begin(), output_ids_map.end(), unk_id);
        if (it != output_ids_map.end() && *it == unk_id)
          output_ids_map.erase(it);
        disable_unk = false;
      }

      if (!output_ids_map.empty()) {
        // Pad vocabulary size to the preferred size multiple.
        while (output_ids_map.size() % preferred_size_multiple() != 0)
          output_ids_map.push_back(0);

        decoder.set_vocabulary_mask(
          StorageView({static_cast<dim_t>(output_ids_map.size())},
                      std::vector<int32_t>(output_ids_map.begin(), output_ids_map.end()),
                      device()));
      } else {
        decoder.reset_vocabulary_mask();
      }

      // Decode.
      auto target_prefix_ids = _target_vocabulary->to_ids(target_prefix_inputs);
      const size_t start_id = _target_vocabulary->to_id(_with_target_bos
                                                        ? Vocabulary::bos_token
                                                        : Vocabulary::eos_token);
      const size_t end_id = _target_vocabulary->to_id(Vocabulary::eos_token);
      const size_t batch_size = source.size();

      std::vector<size_t> start_ids;
      if (_user_decoder_start_tokens)
        start_ids = get_start_ids_from_prefix(target_prefix_ids);
      else
        start_ids.assign(batch_size, start_id);

      std::vector<GenerationResult<size_t>> results = decode(
        decoder,
        state,
        search_strategy,
        sampler,
        start_ids,
        !target_prefix_ids.empty() ? &target_prefix_ids : nullptr,
        !output_ids_map.empty() ? &output_ids_map : nullptr,
        end_id,
        unk_id,
        max_output_length,
        min_output_length,
        num_hypotheses,
        return_alternatives,
        return_scores,
        return_attention || replace_unknowns,
        normalize_scores,
        repetition_penalty,
        disable_unk);

      // Convert generated ids to tokens.
      std::vector<GenerationResult<std::string>> final_results;
      final_results.reserve(results.size());

      for (size_t i = 0; i < batch_size; ++i) {
        GenerationResult<size_t>& result = results[i];
        auto hypotheses = _target_vocabulary->to_tokens(result.hypotheses);

        if (result.has_attention()) {
          // Remove padding and special tokens in attention vectors.
          const size_t offset = size_t(_with_source_bos);
          const size_t source_original_length = source[i].size();
          const size_t source_input_length = source_features[0][i].size();

          for (size_t h = 0; h < result.attention.size(); ++h) {
            auto& attention = result.attention[h];

            for (auto& vector : attention) {
              vector = std::vector<float>(vector.begin() + offset,
                                          vector.begin() + offset + source_input_length);
              vector.resize(source_original_length, 0);
            }

            if (replace_unknowns)
              replace_unknown_tokens(source_features[0][i], hypotheses[h], attention);
          }

          if (!return_attention)
            result.attention.clear();
        }

        final_results.emplace_back(std::move(hypotheses),
                                   std::move(result.scores),
                                   std::move(result.attention));
      }

      return final_results;
    }

  }
}
