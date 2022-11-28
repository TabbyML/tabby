#include "ctranslate2/layers/whisper.h"

namespace ctranslate2 {
  namespace layers {

    WhisperEncoder::WhisperEncoder(const models::Model& model, const std::string& scope)
      : _conv1(model, scope + "/conv1", /*stride=*/1, /*padding=*/1)
      , _conv2(model, scope + "/conv2", /*stride=*/2, /*padding=*/1)
      , _transpose({0, 2, 1})
      , _position_embedding(model, scope + "/position_encodings")
      , _num_heads(model.get_attribute_with_default<int32_t>(scope + "/num_heads", 8))
      , _layers(build_layers_list<const TransformerEncoderLayer>(model,
                                                                 scope + "/layer",
                                                                 _num_heads,
                                                                 /*pre_norm=*/true,
                                                                 ops::ActivationType::GELU))
      , _output_norm(model, scope + "/layer_norm")
    {
    }

    void WhisperEncoder::operator()(const StorageView& features, StorageView& output) {
      PROFILE("WhisperEncoder");

      if (features.rank() != 3)
        throw std::invalid_argument("Expected input features to have 3 dimensions, but got "
                                    + std::to_string(features.rank())
                                    + " dimension(s) instead");

      StorageView input(output_type(), features.device());

      _conv1(features, input);
      _gelu(input, input);

      _conv2(input, output);
      _gelu(output, output);

      _transpose(output, input);
      _position_embedding(input);

      for (const auto& layer : _layers) {
        (*layer)(input, nullptr, output);
        input = std::move(output);
      }

      _output_norm(input, output);
    }


    void WhisperDecoder::forward_prompt(const StorageView& prompt,
                                        DecoderState& state,
                                        dim_t return_logits_at_index,
                                        StorageView* logits) {
      const Device device = prompt.device();
      const DataType dtype = output_type();

      StorageView outputs(dtype, device);
      decode(prompt,
             /*lengths=*/nullptr,
             /*step=*/0,
             state,
             logits ? &outputs : nullptr,
             /*attention=*/nullptr,
             /*return_logits=*/false);

      if (logits) {
        outputs.reshape({-1, outputs.dim(-1)});

        StorageView output_at_index(dtype, device);
        StorageView gather_index({1}, int32_t(return_logits_at_index), device);
        ops::Gather()(outputs, gather_index, output_at_index);

        _proj(output_at_index, *logits);
      }
    }

  }
}
