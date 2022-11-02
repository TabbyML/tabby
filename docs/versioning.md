# Versioning

The project versioning follows [Semantic Versioning 2.0.0](https://semver.org/). The following APIs are covered by backward compatibility guarantees:

* Converted models
* Python converters options
* Python symbols:
  * `ctranslate2.Generator`
  * `ctranslate2.Translator`
  * `ctranslate2.converters.FairseqConverter`
  * `ctranslate2.converters.MarianConverter`
  * `ctranslate2.converters.OpenAIGPT2Converter`
  * `ctranslate2.converters.OpenNMTPyConverter`
  * `ctranslate2.converters.OpenNMTTFConverter`
  * `ctranslate2.converters.OpusMTConverter`
  * `ctranslate2.converters.TransformersConverter`
* C++ symbols:
  * `ctranslate2::TranslationOptions`
  * `ctranslate2::TranslationResult`
  * `ctranslate2::TranslatorPool`
  * `ctranslate2::Translator`
  * `ctranslate2::models::Model`
* C++ translation client options

Other APIs are expected to evolve to increase efficiency, genericity, and model support.
