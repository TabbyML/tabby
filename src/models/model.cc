#include "ctranslate2/models/model.h"

#include <fstream>

#include "ctranslate2/models/transformer.h"
#include "ctranslate2/utils.h"

namespace ctranslate2 {
  namespace models {

    template <typename T>
    T consume(std::istream& in) {
      T val;
      in.read(reinterpret_cast<char*>(&val), sizeof (T));
      return val;
    }

    template <typename T>
    T* consume(std::istream& in, size_t n, T* data = nullptr) {
      if (n == 0)
        return nullptr;
      if (data == nullptr)
        data = new T[n];
      in.read(reinterpret_cast<char*>(data), n * sizeof (T));
      return data;
    }

    template<>
    std::string consume(std::istream& in) {
      auto str_length = consume<uint16_t>(in);
      auto c_str = consume<char>(in, str_length);
      std::string str(c_str);
      delete [] c_str;
      return str;
    }

    static bool endswith(const std::string& str, const std::string& suffix) {
      return (str.size() >= suffix.size() &&
              str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0);
    }


    Model::Model(const std::string& path, size_t spec_revision)
      : _spec_revision(spec_revision) {
      try {
        _shared_vocabulary.reset(new Vocabulary(path + "/shared_vocabulary.txt"));
      } catch (std::exception&) {
        _source_vocabulary.reset(new Vocabulary(path + "/source_vocabulary.txt"));
        _target_vocabulary.reset(new Vocabulary(path + "/target_vocabulary.txt"));
      }
      _vocabulary_map.reset(new VocabularyMap(path + "/vmap.txt", get_target_vocabulary()));
    }

    size_t Model::current_spec_revision() const {
      return 1;
    }

    Device Model::device() const {
      return _device;
    }

    void Model::set_device(Device type, int index) {
      _device = type;
      _device_index = index;
    }

    void Model::set_computType(ComputeType type) {
      _computeType = type;
    }

    ScopedDeviceSetter Model::get_scoped_device_setter() const {
      return ScopedDeviceSetter(_device, _device_index);
    }

    const Vocabulary& Model::get_source_vocabulary() const {
      return _shared_vocabulary ? *_shared_vocabulary : *_source_vocabulary;
    }

    const Vocabulary& Model::get_target_vocabulary() const {
      return _shared_vocabulary ? *_shared_vocabulary : *_target_vocabulary;
    }

    const VocabularyMap& Model::get_vocabulary_map() const {
      return *_vocabulary_map;
    }

    const StorageView* Model::get_variable_if_exists(const std::string& name) const {
      auto alias_it = _variable_alias.find(name);
      auto variable_name = alias_it != _variable_alias.end() ? alias_it->second : name;
      auto it = _variable_index.find(variable_name);
      if (it == _variable_index.end())
        return nullptr;
      return &it->second;
    }

    const StorageView& Model::get_variable(const std::string& name) const {
      const auto* var = get_variable_if_exists(name);
      if (var == nullptr)
        throw std::out_of_range("variable " + name + " not found");
      return *var;
    }

    const std::unordered_map<std::string, StorageView>& Model::get_variables() const {
      return _variable_index;
    }

    void Model::register_variable(const std::string& name, StorageView& variable) {
      _variable_index.emplace(std::piecewise_construct,
                              std::forward_as_tuple(name),
                              std::forward_as_tuple(std::move(variable)));
    }

    void Model::register_variable_alias(const std::string& alias,
                                        const std::string& variable_name) {
      _variable_alias.emplace(alias, variable_name);
      // Also alias the quantization scale that could be associated to variable_name.
      _variable_alias.emplace(alias + "_scale", variable_name + "_scale");
    }

    StorageView* Model::get_scale(const std::string& scale_name, DataType dataType) {
      StorageView *scale = nullptr;
      auto scale_it = _variable_index.find(scale_name);
      if (scale_it != _variable_index.end())
      scale = &scale_it->second;

      // Compatibility with int16 models without a saved scale.
      if (scale == nullptr) {
        if (dataType == DataType::DT_INT16) {
          StorageView compat_scale(ops::Quantize::default_int16_scale);
          Model::register_variable(scale_name, compat_scale);
          scale = &_variable_index.at(scale_name);
        }
      }

      if (!scale) {
        throw std::invalid_argument("Model data is uncompatible. scale is NULL.");
      }

      return scale;
    }

    void
    Model::convert_data_if_need(bool support_int8,
                                bool support_int16,
                                const std::string& name,
                                StorageView& variable,
                                std::vector<std::pair<std::string, StorageView>>& variables_to_add,
                                std::vector<std::string>& variables_to_remove) {
      bool is_int8 = variable.dtype() == DataType::DT_INT8;
      bool is_int16 = variable.dtype() == DataType::DT_INT16;
      bool is_float = variable.dtype() == DataType::DT_FLOAT;

      // Use the same quantization logic as in model_spec.py.
      const ops::Quantize quantize_op(/*int16_scale_type=*/ops::Quantize::ScaleType::PER_LAYER);
      const ops::Dequantize dequantize_op{};

      std::string scale_name = name + "_scale";
      if (_computeType == ComputeType::DEFAULT) {
        if (is_int8 || is_int16) {
          StorageView *scale = get_scale(scale_name, variable.dtype());

          // If quantized variables are not supported, fallback to float32.
          if ((is_int16 && !support_int16) || (is_int8 && !support_int8)) {
            StorageView variable_float;
            dequantize_op(variable, *scale, variable_float);
            swap(variable, variable_float);

            // However, if int16 is supported and we came from int8, quantize to int16.
            if (is_int8 && support_int16) {
              StorageView variable_int16(DataType::DT_INT16);
              quantize_op(variable, variable_int16, *scale);
              swap(variable, variable_int16);
            } else { // is_int16 or !support_int16
              variables_to_remove.emplace_back(std::move(scale_name));
            }
          }
        }
      } else if (_computeType == ComputeType::FLOAT) {

        if (is_float) {
          // do nothing
        } else { // is_int8 || is_int16
          StorageView* scale = get_scale(scale_name, variable.dtype());

          // fallback to float32
          StorageView variable_float;
          dequantize_op(variable, *scale, variable_float);
          swap(variable, variable_float);
        }

      } else if ((_computeType == ComputeType::INT16 && is_int16) ||
                (_computeType == ComputeType::INT8 && is_int8)) {
        // Make & register a scale if the scale is absent
        get_scale(scale_name, variable.dtype());

      } else {
        if (is_float) {

          StorageView scale(DataType::DT_FLOAT);

          // from float32 to int16
          StorageView variable_int(_computeType == ComputeType::INT16 ? DataType::DT_INT16 : DataType::DT_INT8);
          quantize_op(variable, variable_int, scale);
          swap(variable, variable_int);

          variables_to_add.emplace_back(make_pair(scale_name, scale));

        } else { // DataType::DT_INT8 or DataType::DT_INT16
          StorageView *scale = get_scale(scale_name, variable.dtype());

          // from int8 to float32 firstly
          StorageView variable_float;
          dequantize_op(variable, *scale, variable_float);
          swap(variable, variable_float);

          // from float to int
          StorageView variable_int(_computeType == ComputeType::INT8 ? DataType::DT_INT8 : DataType::DT_INT16);
          quantize_op(variable, variable_int, *scale);
          swap(variable, variable_int);
        }
      }
    }

    void Model::finalize() {
      bool support_int8 = mayiuse_int8(_device);
      bool support_int16 = mayiuse_int16(_device);

      std::vector<std::string> variables_to_remove;
      std::vector<std::pair<std::string, StorageView>> variables_to_add;

      // Make sure CPU supports the demanded type
      if ((_computeType == ComputeType::INT8) && (!support_int8)) {
        throw std::invalid_argument("Demanded compute type is int8, but device doesn't support efficient int8 computation.");
      } else if ((_computeType == ComputeType::INT16) && (!support_int16)) {
        throw std::invalid_argument("Demanded compute type is int16, but device doesn't support efficient int16 computation.");
      }

      for (auto& variable_pair : _variable_index) {
        const auto& name = variable_pair.first;
        auto& variable = variable_pair.second;

        // Only process "weight" variables.
        if (endswith(name, "weight")) {
          convert_data_if_need(support_int8,
                               support_int16,
                               name,
                               variable,
                               variables_to_add,
                               variables_to_remove);
        }
      }

      // Remove no longer needed variables.
      for (const auto& name : variables_to_remove)
        _variable_index.erase(name);

      // Add needed variables.
      for (auto& variable_pair : variables_to_add) {
        _variable_index.emplace(std::piecewise_construct,
                                std::forward_as_tuple(std::move(variable_pair.first)),
                                std::forward_as_tuple(std::move(variable_pair.second)));
      }

      // Second pass to move variables on the target device.
      auto scoped_device_setter = get_scoped_device_setter();

      for (auto& pair : _variable_index) {
        auto& variable = pair.second;
        if (!variable.is_scalar() && variable.device() != _device) {
          StorageView variable_device = variable.to(_device);
          swap(variable, variable_device);
        }
      }
    }

    std::shared_ptr<Model> Model::load(const std::string& path,
                                       const std::string& device,
                                       int device_index,
                                       const std::string& computeType) {

      return load(path, str_to_device(device), device_index, str_to_compute_type(computeType));
    }

    std::shared_ptr<Model> Model::load(const std::string& path,
                                       Device device,
                                       int device_index,
                                       ComputeType computeType) {
      std::string model_path = path + "/model.bin";
      std::ifstream model_file(model_path, std::ios_base::in | std::ios_base::binary);
      if (!model_file.is_open())
        throw std::runtime_error("failed to load the model " + model_path);

      // See the model serialization in python/ctranslate2/specs/model_spec.py.
      auto binary_version = consume<uint32_t>(model_file);
      if (binary_version > current_binary_version)
        throw std::runtime_error("unsupported model version "
                                 + std::to_string(binary_version)
                                 + " (latest version supported: "
                                 + std::to_string(current_binary_version)
                                 + ")");

      std::string spec;
      size_t spec_revision;
      if (binary_version >= 2) {
        spec = consume<std::string>(model_file);
        spec_revision = consume<uint32_t>(model_file);
      } else {
        spec_revision = 1;
      }

      Model* model = nullptr;
      if (spec.empty() || spec == "TransformerBase")
        model = new TransformerBaseModel(path, spec_revision);
      else if (spec == "TransformerBig")
        model = new TransformerBigModel(path, spec_revision);
      else
        throw std::invalid_argument("Unsupported model spec " + spec);

      model->set_device(device, device_index);
      model->set_computType(computeType);

      if (spec_revision > model->current_spec_revision())
        throw std::invalid_argument("unsupported " + spec + " revision "
                                    + std::to_string(spec_revision)
                                    + " (latest revision supported: "
                                    + std::to_string(model->current_spec_revision())
                                    + ")");

      auto num_variables = consume<uint32_t>(model_file);
      for (uint32_t i = 0; i < num_variables; ++i) {
        auto name = consume<std::string>(model_file);
        auto rank = consume<uint8_t>(model_file);
        auto dimensions = consume<uint32_t>(model_file, rank);
        auto data_width = consume<uint8_t>(model_file);
        auto data_size = consume<uint32_t>(model_file);

        Shape shape(std::max(static_cast<int>(rank), 1));
        if (rank == 0) {
          shape[0] = 1;
        } else {
          for (unsigned int k = 0; k < rank; k++) {
            shape[k] = static_cast<size_t>(dimensions[k]);
          }
        }

        DataType dtype;
        switch (data_width) {
        case 4:
          dtype = DataType::DT_FLOAT;
          break;
        case 2:
          dtype = DataType::DT_INT16;
          break;
        case 1:
          dtype = DataType::DT_INT8;
          break;
        default:
          throw std::runtime_error("unsupported data type");
        }

        StorageView variable(shape, dtype);
        consume<char>(model_file, data_size * data_width, static_cast<char*>(variable.buffer()));
        model->register_variable(name, variable);

        delete [] dimensions;
      }

      if (binary_version >= 3) {
        auto num_aliases = consume<uint32_t>(model_file);
        for (uint32_t i = 0; i < num_aliases; ++i) {
          auto alias = consume<std::string>(model_file);
          auto variable_name = consume<std::string>(model_file);
          model->register_variable_alias(alias, variable_name);
        }
      }

      model->finalize();
      return std::shared_ptr<Model>(model);
    }

  }
}
