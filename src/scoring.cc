#include "ctranslate2/scoring.h"

namespace ctranslate2 {

  std::vector<ScoringResult>
  score_sequences(layers::Decoder& decoder,
                  layers::DecoderState& state,
                  const std::vector<std::vector<size_t>>& sequences,
                  const Vocabulary& vocabulary,
                  const dim_t preferred_size_multiple,
                  const dim_t offset) {
    const dim_t batch_size = sequences.size();
    const Device device = decoder.device();

    std::vector<std::vector<size_t>> input_sequences;
    std::vector<std::vector<size_t>> output_sequences;
    input_sequences.reserve(batch_size);
    output_sequences.reserve(batch_size);

    for (dim_t b = 0; b < batch_size; ++b) {
      auto& sequence = sequences[b];
      if (sequence.empty()) {
        input_sequences.emplace_back();
        output_sequences.emplace_back();
      } else {
        input_sequences.emplace_back(sequence.begin(), sequence.end() - 1);
        output_sequences.emplace_back(sequence.begin() + 1, sequence.end());
      }
    }

    StorageView lengths;
    const StorageView input_ids = layers::make_sequence_inputs(input_sequences,
                                                               device,
                                                               preferred_size_multiple,
                                                               &lengths);
    const StorageView output_ids = layers::make_sequence_inputs(output_sequences,
                                                                device,
                                                                preferred_size_multiple);

    decoder.update_output_layer(preferred_size_multiple);

    StorageView logits(decoder.output_type(), device);
    decoder(input_ids, lengths, state, logits);
    ops::LogSoftMax()(logits);
    StorageView log_probs = std::move(logits);

    StorageView scores(log_probs.dtype(), device);
    ops::Gather(/*axis=*/-1, /*batch_dims=*/2)(log_probs, output_ids, scores);

    if (scores.device() != Device::CPU)
      scores = scores.to(Device::CPU);
    if (scores.dtype() != DataType::FLOAT32)
      scores = scores.to_float32();

    std::vector<ScoringResult> results(batch_size);
    for (dim_t b = 0; b < batch_size; ++b) {
      const dim_t output_length = output_sequences[b].size();
      auto& result = results[b];
      result.tokens.reserve(output_length);
      result.tokens_score.reserve(output_length);
      for (dim_t t = offset; t < output_length; ++t) {
        result.tokens.emplace_back(vocabulary.to_token(output_sequences[b][t]));
        result.tokens_score.emplace_back(scores.at<float>({b, t}));
      }
    }

    return results;
  }

}
