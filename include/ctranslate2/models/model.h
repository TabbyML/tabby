#pragma once

#include <unordered_map>
#include <memory>

#include "ctranslate2/vocabulary.h"
#include "ctranslate2/vocabulary_map.h"
#include "ctranslate2/layers/encoder.h"
#include "ctranslate2/layers/decoder.h"

namespace ctranslate2 {
  namespace models {

    // Base class for models.
    class Model {
    public:
      Model(const std::string& path, size_t spec_revision);
      virtual ~Model() = default;
      virtual size_t current_spec_revision() const;

      Device device() const;
      void set_device(Device type, int index = 0);
      void set_computType(ComputeType type);
      ScopedDeviceSetter get_scoped_device_setter() const;

      const Vocabulary& get_source_vocabulary() const;
      const Vocabulary& get_target_vocabulary() const;
      const VocabularyMap& get_vocabulary_map() const;

      const StorageView* get_variable_if_exists(const std::string& name) const;
      const StorageView& get_variable(const std::string& name) const;
      const std::unordered_map<std::string, StorageView>& get_variables() const;

      // Models can override these methods to execute some transformations if needed
      // (e.g. a variable name changed in a newer spec revision).
      virtual void register_variable(const std::string& name, StorageView& variable);
      StorageView* get_scale(const std::string& scale_name, DataType dataType);
      virtual void finalize();

      // Makes new graph to execute this model. Graphs returned by these function
      // should support being executed in parallel without duplicating the model
      // data (i.e. the weights).
      virtual std::unique_ptr<layers::Encoder> make_encoder() const = 0;
      virtual std::unique_ptr<layers::Decoder> make_decoder() const = 0;

    protected:
      Device _device;
      int _device_index;
      const Vocabulary _source_vocabulary;
      const Vocabulary _target_vocabulary;
      const VocabularyMap _vocabulary_map;
      std::unordered_map<std::string, StorageView> _variable_index;
      size_t _spec_revision;
      ComputeType _computeType = ComputeType::DEFAULT;

      void convert_data_if_need(bool support_int8, bool support_int16, std::pair<const std::basic_string<char>, StorageView>& variable_pair, std::vector<std::string>& variables_to_remove);
    };


    static const size_t current_binary_version = 2;

    // Model factory from a path.
    class ModelFactory {
    public:
      static std::shared_ptr<Model> load(const std::string& path,
                                         Device device,
                                         int device_index = 0,
                                         ComputeType computeType = ComputeType::DEFAULT);
    };

  }
}
