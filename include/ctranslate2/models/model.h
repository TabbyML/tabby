#pragma once

#include <unordered_map>
#include <memory>

#include "ctranslate2/models/model_reader.h"
#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace models {

    static const size_t current_binary_version = 5;

    // Checks whether the provided path could contain a CTranslate2 model.
    bool contains_model(const std::string& path);

    class SequenceToSequenceReplica;
    class SequenceGeneratorReplica;

    // Base class for models.
    class Model : public std::enable_shared_from_this<Model> {
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

      virtual std::unique_ptr<SequenceToSequenceReplica> as_sequence_to_sequence() const;
      virtual std::unique_ptr<SequenceGeneratorReplica> as_sequence_generator() const;

      virtual ~Model();

      size_t binary_version() const {
        return _binary_version;
      }

      size_t spec_revision() const {
        return _spec_revision;
      }

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

      bool round_before_cast_in_quantization() const {
        return _binary_version >= 5;
      }

      ScopedDeviceSetter get_scoped_device_setter() const {
        return ScopedDeviceSetter(_device, _device_index);
      }

      // If the model contains variables, they will be moved to the new device.
      void set_device(const Device device, const int index = 0);

      const StorageView* get_variable_if_exists(const std::string& name) const;
      const StorageView& get_variable(const std::string& name) const;
      std::unordered_map<std::string, StorageView> get_variables() const;
      bool layer_exists(std::string prefix) const;

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
      virtual void register_variable(std::string name, StorageView variable);
      virtual void register_variable_alias(std::string alias, std::string variable_name);
      virtual void remove_variable(const std::string& name);

      // Runs some initialization after the model is loaded.
      virtual void initialize(ModelReader&) {}

    private:
      void process_linear_weights();
      void set_compute_type(ComputeType type, Device device, int device_index);
      void ensure_dtype(const std::string& name,
                        StorageView& variable,
                        const DataType target_dtype);
      ComputeType infer_compute_type() const;

      Device _device = Device::CPU;
      int _device_index = 0;
      size_t _binary_version = 0;
      size_t _spec_revision = 0;
      ComputeType _compute_type = ComputeType::DEFAULT;
      ComputeType _effective_compute_type = ComputeType::DEFAULT;
      dim_t _preferred_size_multiple = 1;
      std::unordered_map<std::string, std::shared_ptr<StorageView>> _variable_index;
    };

    template<>
    inline std::string Model::get_attribute_with_default(const std::string& name,
                                                         std::string default_value) const {
      const StorageView* attribute = get_variable_if_exists(name);
      if (!attribute)
        return default_value;
      StorageView attribute_host = attribute->to(Device::CPU);
      return std::string(reinterpret_cast<const char*>(attribute_host.data<int8_t>()),
                         attribute_host.size());
    }

    // Load a model replica on each device ID configured in device_indices.
    // Replicas on the same device ID will reference the same model instance.
    std::vector<std::shared_ptr<const Model>>
    load_replicas(models::ModelReader& model_reader,
                  const Device device,
                  const std::vector<int>& device_indices,
                  const ComputeType compute_type);

    std::vector<std::shared_ptr<const Model>>
    load_replicas(const std::string& model_path,
                  const Device device,
                  const std::vector<int>& device_indices,
                  const ComputeType compute_type);

    std::vector<std::shared_ptr<const Model>>
    load_replicas(models::ModelReader& model_reader,
                  const Device device,
                  const std::vector<int>& device_indices,
                  const ComputeType compute_type,
                  const size_t num_replicas_per_device);

    // Base class for replicas.
    // A replica colocates runtime resources with a model instance.
    class ModelReplica {
    public:
      virtual ~ModelReplica() = default;

      ModelReplica(const std::shared_ptr<const Model>& model)
        : _model(model)
      {
      }

      const std::shared_ptr<const Model>& model() const {
        return _model;
      }

    private:
      const std::shared_ptr<const Model> _model;
    };

  }
}
