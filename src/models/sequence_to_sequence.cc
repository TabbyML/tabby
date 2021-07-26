#include "ctranslate2/models/sequence_to_sequence.h"

#include <numeric>

namespace ctranslate2 {
  namespace models {

    static const std::string shared_vocabulary_file = "shared_vocabulary.txt";
    static const std::string source_vocabulary_file = "source_vocabulary.txt";
    static const std::string target_vocabulary_file = "target_vocabulary.txt";
    static const std::string vmap_file = "vmap.txt";

    SequenceToSequenceModel::SequenceToSequenceModel(ModelReader& model_reader, size_t spec_revision)
      : Model(model_reader, spec_revision) {
      {
        auto shared_vocabulary = model_reader.get_file(shared_vocabulary_file);
        if (shared_vocabulary) {
          _source_vocabulary = std::make_shared<Vocabulary>(*shared_vocabulary);
          _target_vocabulary = _source_vocabulary;
        } else {
          {
            auto source_vocabulary = model_reader.get_required_file(source_vocabulary_file);
            _source_vocabulary = std::make_shared<Vocabulary>(*source_vocabulary);
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
    }

    const Vocabulary& SequenceToSequenceModel::get_source_vocabulary() const {
      return *_source_vocabulary;
    }

    const Vocabulary& SequenceToSequenceModel::get_target_vocabulary() const {
      return *_target_vocabulary;
    }

    const VocabularyMap* SequenceToSequenceModel::get_vocabulary_map() const {
      return _vocabulary_map && !_vocabulary_map->empty() ? _vocabulary_map.get() : nullptr;
    }

    void SequenceToSequenceModel::forward_encoder(layers::Encoder& encoder,
                                                  const std::vector<std::vector<std::string>>& source,
                                                  StorageView& memory,
                                                  StorageView& memory_lengths) const {
      const auto scoped_device_setter = get_scoped_device_setter();
      PROFILE("SequenceToSequenceModel::forward_encoder");
      const auto source_ids = _source_vocabulary->to_ids(source,
                                                         _with_source_bos,
                                                         _with_source_eos);

      StorageView ids;
      std::tie(ids, memory_lengths) = layers::make_sequence_inputs(source_ids,
                                                                   _device,
                                                                   _preferred_size_multiple);

      encoder(ids, memory_lengths, memory);
    }

    void SequenceToSequenceModel::forward_decoder(layers::Decoder& decoder,
                                                  layers::DecoderState& state,
                                                  const std::vector<std::vector<std::string>>& target,
                                                  StorageView& log_probs) const {
      const auto scoped_device_setter = get_scoped_device_setter();
      PROFILE("SequenceToSequenceModel::forward_decoder");
      const auto target_ids = _target_vocabulary->to_ids(target,
                                                         /*add_bos=*/true,
                                                         /*add_eos=*/false);

      StorageView ids;
      StorageView lengths;
      std::tie(ids, lengths) = layers::make_sequence_inputs(target_ids,
                                                            _device,
                                                            _preferred_size_multiple);


      StorageView logits(decoder.output_type(), _device);
      decoder(ids, lengths, state, logits);
      ops::LogSoftMax()(logits, log_probs);
    }

    void SequenceToSequenceModel::forward(layers::Encoder& encoder,
                                          layers::Decoder& decoder,
                                          const std::vector<std::vector<std::string>>& source,
                                          const std::vector<std::vector<std::string>>& target,
                                          StorageView& log_probs) const {
      const auto scoped_device_setter = get_scoped_device_setter();
      PROFILE("SequenceToSequenceModel::forward");
      StorageView memory(encoder.output_type(), _device);
      StorageView memory_lengths(DataType::INT32, _device);
      forward_encoder(encoder, source, memory, memory_lengths);

      layers::DecoderState state = decoder.initial_state(/*iterative_decoding=*/false);
      state.emplace("memory", std::move(memory));
      state.emplace("memory_lengths", std::move(memory_lengths));
      forward_decoder(decoder, state, target, log_probs);
    }

    std::vector<ScoringResult>
    SequenceToSequenceModel::score(layers::Encoder& encoder,
                                   layers::Decoder& decoder,
                                   const std::vector<std::vector<std::string>>& source,
                                   const std::vector<std::vector<std::string>>& target) const {
      const auto scoped_device_setter = get_scoped_device_setter();
      PROFILE("SequenceToSequenceModel::score");
      StorageView log_probs(decoder.output_type(), _device);
      forward(encoder, decoder, source, target, log_probs);

      const auto target_ids_out = _target_vocabulary->to_ids(target,
                                                             /*add_bos=*/false,
                                                             /*add_eos=*/true);

      StorageView gather_ids;
      std::tie(gather_ids, std::ignore) = layers::make_sequence_inputs(target_ids_out,
                                                                       _device,
                                                                       _preferred_size_multiple);

      StorageView scores(log_probs.dtype(), _device);
      ops::Gather(/*axis=*/-1, /*batch_dims=*/2)(log_probs, gather_ids, scores);

      if (scores.device() != Device::CPU)
        scores = scores.to(Device::CPU);
      if (scores.dtype() != DataType::FLOAT)
        scores = scores.to_float();

      const dim_t batch_size = scores.dim(0);
      std::vector<ScoringResult> results(batch_size);
      for (dim_t b = 0; b < batch_size; ++b) {
        const dim_t max_time = target[b].size();
        auto& result = results[b];
        result.tokens = target[b];
        result.tokens_score.resize(max_time);
        for (dim_t t = 0; t < max_time; ++t)
          result.tokens_score[t] = scores.at<float>({b, t});
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

    std::vector<GenerationResult<std::string>>
    SequenceToSequenceModel::sample(layers::Encoder& encoder,
                                    layers::Decoder& decoder,
                                    const std::vector<std::vector<std::string>>& source,
                                    const std::vector<std::vector<std::string>>& target_prefix,
                                    const SearchStrategy& search_strategy,
                                    const Sampler& sampler,
                                    const bool use_vmap,
                                    const size_t max_length,
                                    const size_t min_length,
                                    const size_t num_hypotheses,
                                    const bool return_alternatives,
                                    const bool return_scores,
                                    const bool return_attention,
                                    const bool replace_unknowns,
                                    const bool normalize_scores) const {
      const auto scoped_device_setter = get_scoped_device_setter();
      PROFILE("SequenceToSequenceModel::sample");

      // Encode the sequence.
      StorageView memory(encoder.output_type(), _device);
      StorageView memory_lengths(DataType::INT32, _device);
      forward_encoder(encoder, source, memory, memory_lengths);

      layers::DecoderState state = decoder.initial_state();
      state.emplace("memory", std::move(memory));
      state.emplace("memory_lengths", std::move(memory_lengths));

      std::vector<size_t> output_ids_map;
      if (use_vmap && _vocabulary_map) {
        output_ids_map = _vocabulary_map->get_candidates(source);
      } else if (_target_vocabulary->size() % _preferred_size_multiple != 0) {
        output_ids_map.resize(_target_vocabulary->size());
        std::iota(output_ids_map.begin(), output_ids_map.end(), size_t(0));
      }

      if (!output_ids_map.empty()) {
        // Pad vocabulary size to the preferred size multiple.
        while (output_ids_map.size() % _preferred_size_multiple != 0)
          output_ids_map.push_back(0);

        decoder.set_vocabulary_mask(
          StorageView({static_cast<dim_t>(output_ids_map.size())},
                      std::vector<int32_t>(output_ids_map.begin(), output_ids_map.end()),
                      _device));
      } else {
        decoder.reset_vocabulary_mask();
      }

      // Decode.
      const auto target_prefix_ids = _target_vocabulary->to_ids(target_prefix);
      const size_t start_id = _target_vocabulary->to_id(_with_target_bos
                                                        ? Vocabulary::bos_token
                                                        : Vocabulary::eos_token);
      const size_t end_id = _target_vocabulary->to_id(Vocabulary::eos_token);
      const size_t batch_size = source.size();
      const std::vector<size_t> start_ids(batch_size, start_id);
      std::vector<GenerationResult<size_t>> results = decode(
        decoder,
        state,
        search_strategy,
        sampler,
        start_ids,
        !target_prefix_ids.empty() ? &target_prefix_ids : nullptr,
        !output_ids_map.empty() ? &output_ids_map : nullptr,
        end_id,
        max_length,
        min_length,
        num_hypotheses,
        return_alternatives,
        return_scores,
        return_attention || replace_unknowns,
        normalize_scores);

      // Convert generated ids to tokens.
      std::vector<GenerationResult<std::string>> final_results;
      final_results.reserve(results.size());

      for (size_t i = 0; i < batch_size; ++i) {
        GenerationResult<size_t>& result = results[i];
        auto hypotheses = _target_vocabulary->to_tokens(result.hypotheses);

        if (result.has_attention()) {
          // Remove padding and special tokens in attention vectors.
          const size_t offset = size_t(_with_source_bos);
          const size_t length = source[i].size();

          for (size_t h = 0; h < result.attention.size(); ++h) {
            auto& attention = result.attention[h];

            for (auto& vector : attention) {
              vector = std::vector<float>(vector.begin() + offset, vector.begin() + offset + length);
            }

            if (replace_unknowns)
              replace_unknown_tokens(source[i], hypotheses[h], attention);
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
