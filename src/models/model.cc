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

    static std::string consume_string(std::istream& in) {
      auto str_length = consume<uint16_t>(in);
      auto c_str = consume<char>(in, str_length);
      std::string str(c_str);
      delete [] c_str;
      return str;
    }


    Model::Model(const std::string& path, size_t spec_revision)
      : _source_vocabulary(path + "/source_vocabulary.txt")
      , _target_vocabulary(path + "/target_vocabulary.txt")
      , _vocabulary_map(path + "/vmap.txt", _target_vocabulary)
      , _spec_revision(spec_revision) {
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

    void Model::use_model_device() const {
      DEVICE_DISPATCH(_device, primitives<D>::set_device(_device_index));
    }

    const Vocabulary& Model::get_source_vocabulary() const {
      return _source_vocabulary;
    }

    const Vocabulary& Model::get_target_vocabulary() const {
      return _target_vocabulary;
    }

    const VocabularyMap& Model::get_vocabulary_map() const {
      return _vocabulary_map;
    }

    const StorageView* Model::get_variable_if_exists(const std::string& name) const {
      auto it = _variable_index.find(name);
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

    void Model::register_variable(const std::string& name, StorageView& variable) {
      _variable_index.emplace(std::piecewise_construct,
                              std::forward_as_tuple(name),
                              std::forward_as_tuple(std::move(variable)));
    }

    void Model::finalize() {
      bool support_int8 = mayiuse_int8(_device);
      bool support_int16 = mayiuse_int16(_device);

      static const StorageView default_int16_scale(static_cast<float>(1000));
      std::vector<std::string> variables_to_remove;

      // First pass to possibly cast to a supported type.
      for (auto& pair : _variable_index) {
        const auto& name = pair.first;
        auto& variable = pair.second;

        bool is_int8 = variable.dtype() == DataType::DT_INT8;
        bool is_int16 = variable.dtype() == DataType::DT_INT16;

        if (is_int8 || is_int16) {
          StorageView* scale = nullptr;
          std::string scale_name = name + "_scale";
          auto scale_it = _variable_index.find(scale_name);
          if (scale_it != _variable_index.end())
            scale = &scale_it->second;

          // Compatibility with int16 models without a saved scale.
          if (is_int16 && scale == nullptr) {
            StorageView compat_scale(default_int16_scale);
            Model::register_variable(scale_name, compat_scale);
            scale = &_variable_index.at(scale_name);
          }

          // If quantized variables are not supported, fallback to float32.
          if ((is_int16 && !support_int16) || (is_int8 && !support_int8)) {
            StorageView variable_float;
            ops::Unquantize()(variable, *scale, variable_float);
            swap(variable, variable_float);

            // However, if int16 is supported and we came from int8, quantize to int16.
            if (is_int8 && support_int16) {
              *scale = default_int16_scale;
              StorageView variable_int16(DataType::DT_INT16);
              ops::Quantize()(variable, *scale, variable_int16);
              swap(variable, variable_int16);
            } else {
              variables_to_remove.emplace_back(std::move(scale_name));
            }
          }
        }
      }

      // Remove no longer needed variables.
      for (const auto& name : variables_to_remove)
        _variable_index.erase(name);

      // Second pass to move variables on the target device.
      for (auto& pair : _variable_index) {
        auto& variable = pair.second;
        if (!variable.is_scalar() && variable.device() != _device) {
          StorageView variable_device = variable.to(_device);
          swap(variable, variable_device);
        }
      }
    }

    std::shared_ptr<Model> ModelFactory::load(const std::string& path,
                                              Device device,
                                              int device_index) {
      std::string model_path = path + "/model.bin";
      std::ifstream model_file(model_path, std::ios_base::in | std::ios_base::binary);
      if (!model_file.is_open())
        throw std::runtime_error("failed to load the model " + model_path);

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
        spec = consume_string(model_file);
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
      model->use_model_device();

      if (spec_revision > model->current_spec_revision())
        throw std::invalid_argument("unsupported " + spec + " revision "
                                    + std::to_string(spec_revision)
                                    + " (latest revision supported: "
                                    + std::to_string(model->current_spec_revision())
                                    + ")");

      auto num_variables = consume<uint32_t>(model_file);
      for (uint32_t i = 0; i < num_variables; ++i) {
        auto name = consume_string(model_file);
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

      model->finalize();
      return std::shared_ptr<Model>(model);
    }

  }
}
