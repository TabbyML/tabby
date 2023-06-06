# Versioning

The project versioning follows [Semantic Versioning 2.0.0](https://semver.org/). The following APIs are covered by backward compatibility guarantees:

* Converted models
* Python converters options
* Python symbols:
  * `ctranslate2.*`
  * `ctranslate2.converters.*`
* C++ symbols:
  * `ctranslate2::Encoder`
  * `ctranslate2::EncoderForwardOutput`
  * `ctranslate2::GenerationOptions`
  * `ctranslate2::GenerationResult`
  * `ctranslate2::Generator`
  * `ctranslate2::ScoringOptions`
  * `ctranslate2::ScoringResult`
  * `ctranslate2::TranslationOptions`
  * `ctranslate2::TranslationResult`
  * `ctranslate2::Translator`
  * `ctranslate2::models::Model::load`
* C++ translation client options

Other APIs are expected to evolve to increase efficiency, genericity, and model support.
