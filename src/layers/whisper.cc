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

      if (features.dim(1) != input_size() || features.dim(2) > max_input_time())
        throw std::invalid_argument("Invalid input features shape: expected an input with shape ("
                                    + std::to_string(features.dim(0))
                                    + ", "
                                    + std::to_string(input_size())
                                    + ", "
                                    + std::to_string(std::min(features.dim(2), max_input_time()))
                                    + "), but got an input with shape ("
                                    + std::to_string(features.dim(0))
                                    + ", "
                                    + std::to_string(features.dim(1))
                                    + ", "
                                    + std::to_string(features.dim(2))
                                    + ") instead");

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
                                        StorageView* outputs) {
      decode(prompt,
             /*lengths=*/nullptr,
             /*step=*/0,
             state,
             outputs,
             /*attention=*/nullptr,
             /*return_logits=*/false);
    }

    void WhisperDecoder::compute_logits_for_steps(const StorageView& outputs,
                                                  const StorageView& steps,
                                                  StorageView& logits) {
      StorageView step_outputs(outputs.dtype(), outputs.device());
      ops::Gather(1, 1)(outputs, steps, step_outputs);
      _proj(step_outputs, logits);
    }

  }
}
