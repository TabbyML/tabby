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
    T* consume(std::istream& in, size_t n) {
      if (n == 0)
        return nullptr;
      T* data = new T[n];
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


    Model::Model(const std::string& path, size_t spec_revision, Device device)
      : _device(device)
      , _source_vocabulary(path + "/source_vocabulary.txt")
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
      // Finalize variables: possibly cast to a supported type and move to the target device.
      for (auto& pair : _variable_index) {
        const auto& name = pair.first;
        auto& variable = pair.second;

        if (variable.dtype() == DataType::DT_INT16) {
          std::string scale_name = name + "_scale";
          const auto* scale = get_variable_if_exists(scale_name);
          if (scale == nullptr) {
            // Compatibility with models without a saved scale.
            StorageView compat_scale(static_cast<float>(1000));
            Model::register_variable(scale_name, compat_scale);
            scale = get_variable_if_exists(scale_name);
          }

          // Cast int16 back to float on GPU or when AVX2 is not supported.
          if (_device == Device::CUDA || !support_avx2()) {
            StorageView variable_cast;
            ops::Unquantize()(variable, *scale, variable_cast);
            swap(variable, variable_cast);
          }
        }

        if (!variable.is_scalar() && variable.device() != _device) {
          // Move variable on device.
          StorageView variable_device = variable.to(_device);
          swap(variable, variable_device);
        }
      }
    }

    static StorageView load_storage(void* data, size_t data_width, const Shape& shape) {
      std::unique_ptr<StorageView> view;  // No copy view.

      if (data_width == 4) {
        view.reset(new StorageView(shape, reinterpret_cast<float*>(data)));
      } else if (data_width == 2) {
        view.reset(new StorageView(shape, reinterpret_cast<int16_t*>(data)));
      } else if (data_width == 1) {
        view.reset(new StorageView(shape, reinterpret_cast<int8_t*>(data)));
      } else {
        throw std::runtime_error("unsupported data type");
      }

      // Return a copy so that the storage owns and aligns the data.
      return StorageView(*view);
    }

    std::shared_ptr<Model> ModelFactory::load(const std::string& path, Device device) {
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
        model = new TransformerBaseModel(path, spec_revision, device);
      else if (spec == "TransformerBig")
        model = new TransformerBigModel(path, spec_revision, device);
      else
        throw std::invalid_argument("Unsupported model spec " + spec);

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
        auto data = consume<char>(model_file, data_size * data_width);

        std::vector<size_t> shape(std::max(static_cast<int>(rank), 1));
        if (rank == 0) {
          shape[0] = 1;
        } else {
          for (unsigned int k = 0; k < rank; k++) {
            shape[k] = static_cast<size_t>(dimensions[k]);
          }
        }

        auto storage = load_storage(data, data_width, shape);
        model->register_variable(name, storage);

        delete [] dimensions;
        delete [] data;
      }

      model->finalize();
      return std::shared_ptr<Model>(model);
    }

  }
}
