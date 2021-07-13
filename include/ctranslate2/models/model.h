#pragma once

#include <istream>
#include <unordered_map>
#include <memory>

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace models {

    static const size_t current_binary_version = 4;

    // Checks whether the provided path could contain a CTranslate2 model.
    bool contains_model(const std::string& path);

    class ModelReader;

    // Base class for models.
    class Model {
    public:
      static std::shared_ptr<const Model> load(const std::string& path,
                                               Device device = Device::CPU,
                                               int device_index = 0,
                                               ComputeType compute_type = ComputeType::DEFAULT);
      static std::shared_ptr<const Model> load(ModelReader& model_reader,
                                               Device device = Device::CPU,
                                               int device_index = 0,
                                               ComputeType compute_type = ComputeType::DEFAULT);
      static std::shared_ptr<const Model> load(const std::string& path,
                                               const std::string& device,
                                               int device_index,
                                               const std::string& compute_type);

      virtual ~Model() = default;
      virtual size_t current_spec_revision() const;

      Device device() const {
        return _device;
      }

      int device_index() const {
        return _device_index;
      }

      // The requested compute type.
      ComputeType compute_type() const {
        return _compute_type;
      }

      // The compute type that is effectively used.
      ComputeType effective_compute_type() const {
        return _effective_compute_type;
      }

      dim_t preferred_size_multiple() const {
        return _preferred_size_multiple;
      }

      ScopedDeviceSetter get_scoped_device_setter() const {
        return ScopedDeviceSetter(_device, _device_index);
      }

      // If the model contains variables, they will be moved to the new device.
      void set_device(const Device device, const int index = 0);

      const StorageView* get_variable_if_exists(const std::string& name) const;
      const StorageView& get_variable(const std::string& name) const;
      const std::unordered_map<std::string, StorageView>& get_variables() const;

      // Attributes are saved as scalar variables.
      template <typename T>
      T get_attribute_with_default(const std::string& name, T default_value) const {
        const StorageView* attribute = get_variable_if_exists(name);
        if (!attribute)
          return default_value;
        return attribute->as_scalar<T>();
      }

      // A flag is a boolean attribute.
      bool get_flag_with_default(const std::string& name, bool default_value) const;

    protected:
      Model(ModelReader& model_reader, size_t spec_revision);

      // Returns true if the variable is quantizable and should respect compute_type.
      virtual bool is_quantizable(const std::string& variable_name) const;

      // Returns true if the variable is the weight of a linear/dense layer.
      virtual bool is_linear_weight(const std::string& variable_name) const;

      // Returns true if the variable can be pre-packed.
      virtual bool is_packable(const std::string& variable_name) const;

      // Returns true if the variable can be converted to another type.
      virtual bool is_convertible(const StorageView& variable, const std::string& name) const;

      // Models can override these methods to execute some transformations if needed
      // (e.g. a variable name changed in a newer spec revision).
      virtual void register_variable(const std::string& name, StorageView& variable);
      virtual void register_variable_alias(const std::string& alias,
                                           const std::string& variable_name);
      virtual void finalize();

      Device _device;
      int _device_index;
      std::unordered_map<std::string, StorageView> _variable_index;
      size_t _spec_revision;
      ComputeType _compute_type = ComputeType::DEFAULT;
      ComputeType _effective_compute_type = ComputeType::DEFAULT;
      dim_t _preferred_size_multiple = 1;

    private:
      void process_linear_weights();
      void set_compute_type(ComputeType type);
      void ensure_dtype(const std::string& name,
                        StorageView& variable,
                        const DataType target_dtype,
                        std::unordered_map<std::string, StorageView>& variables_to_add,
                        std::vector<std::string>& variables_to_remove);
      ComputeType infer_compute_type() const;
    };

    // The ModelReader interface allows user code to customize how and where to read model files.
    class ModelReader {
    public:
      virtual ~ModelReader() = default;

      // Returns a string identifying the model to be loaded (e.g. its path on disk).
      virtual std::string get_model_id() const = 0;
      // Returns a stream over a file included in the model, or nullptr if the file can't be openned.
      virtual std::unique_ptr<std::istream> get_file(const std::string& filename,
                                                     const bool binary = false) = 0;

      // Wrapper around get_file, raises an exception if the file can't be openned.
      std::unique_ptr<std::istream> get_required_file(const std::string& filename,
                                                      const bool binary = false);

    };

    class ModelFileReader : public ModelReader {
    public:
      ModelFileReader(std::string model_dir, std::string path_separator = "/");
      std::string get_model_id() const override;
      std::unique_ptr<std::istream> get_file(const std::string& filename,
                                             const bool binary = false) override;

    private:
      std::string _model_dir;
      std::string _path_separator;
    };

    // Load a model replica on each device ID configured in device_indices.
    // Replicas on the same device ID will reference the same model instance.
    std::vector<std::shared_ptr<const Model>>
    load_replicas(const std::string& model_path,
                  const Device device,
                  const std::vector<int>& device_indices,
                  const ComputeType compute_type);

  }
}
