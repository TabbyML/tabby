#include "ctranslate2/layers/wav2vec2.h"


namespace ctranslate2 {
  namespace layers {
    Wav2Vec2Encoder::Wav2Vec2Encoder(const models::Model& model, const std::string& scope)
      : _num_heads(model.get_attribute_with_default<int32_t>(scope + "/num_heads", 8))
      , _layers(build_layers_list<const TransformerEncoderLayer>(model,
                                                                 scope + "/layer",
                                                                 _num_heads,
                                                                 /*pre_norm=*/true,
                                                                 ops::ActivationType::GELU))
      , _output_norm(model, scope + "/layer_norm")
    {
    }

    void Wav2Vec2Encoder::operator()(const StorageView& features, StorageView& output) {
      PROFILE("Wav2Vec2Encoder");

      // SAD in front-end handles the input length
      //const dim_t expected_depth = 1024;
      //const dim_t expected_time = 406;

      if (features.rank() != 3)
        throw std::invalid_argument("Expected input features to have 3 dimensions, but got "
                                    + std::to_string(features.rank())
                                    + " dimension(s) instead");
      /* //may need to limit the input lenght
      if (features.dim(1) != expected_depth || features.dim(2) != expected_time)
        throw std::invalid_argument("Invalid input features shape: expected an input with shape ("
                                    + std::to_string(features.dim(0))
                                    + ", "
                                    + std::to_string(expected_depth)
                                    + ", "
                                    + std::to_string(expected_time)
                                    + "), but got an input with shape ("
                                    + std::to_string(features.dim(0))
                                    + ", "
                                    + std::to_string(features.dim(1))
                                    + ", "
                                    + std::to_string(features.dim(2))
                                    + ") instead;; _conv1.output_size() "
                                    + std::to_string(_conv1.output_size()));
                                    //+ ") instead");
      */

      StorageView input(output_type(), features.device());
      input = features;
      for (const auto& layer : _layers) {
        (*layer)(input, nullptr, output);
        input = std::move(output);
      }

      _output_norm(input, output);
    }

  }
}
