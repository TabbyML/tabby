#include "ctranslate2/models/model.h"

#include <spdlog/spdlog.h>

#include "ctranslate2/models/model_factory.h"
#include "ctranslate2/ops/ops.h"
#include "ctranslate2/utils.h"

#ifdef CT2_WITH_CUDA
#  include "cuda/utils.h"
#endif

#include "cpu/backend.h"

namespace ctranslate2 {
  namespace models {

    static const std::string binary_file = "model.bin";
    static const std::string config_file = "config.json";

    static inline void report_stream_error(const std::streampos position,
                                           const size_t read_size,
                                           const std::string& read_type) {
      throw std::runtime_error("File " + binary_file + " is incomplete: "
                               + "failed to read a " + read_type + " of size "
                               + std::to_string(read_size)
                               + " at position "
                               + std::to_string(position));
    }

    template <typename T>
    T consume(std::istream& in) {
      const std::streampos position = in.tellg();
      const size_t read_size = sizeof (T);
      T val;
      in.read(reinterpret_cast<char*>(&val), read_size);
      if (!in)
        report_stream_error(position, read_size, "value");
      return val;
    }

    template <typename T>
    T* consume(std::istream& in, size_t n, T* data = nullptr) {
      if (n == 0)
        return nullptr;
      const std::streampos position = in.tellg();
      const size_t read_size = n * sizeof (T);
      T* dst = data ? data : new T[n];
      in.read(reinterpret_cast<char*>(dst), read_size);
      if (!in) {
        if (dst != data)
          delete [] dst;
        report_stream_error(position, read_size, "buffer");
      }
      return dst;
    }

    template<>
    std::string consume(std::istream& in) {
      const auto str_length = consume<uint16_t>(in);
      const auto c_str = consume<char>(in, str_length);
      std::string str(c_str);
      delete [] c_str;
      return str;
    }

    template <typename VariablesCollection>
    static void move_variables_to_device(VariablesCollection& variables, const Device device) {
      for (auto& pair : variables) {
        StorageView& variable = *pair.second;
        if (variable.is_scalar() || variable.device() == device)
          continue;
        variable = variable.to(device);
      }
    }

    template <typename VariablesCollection>
    static void move_variables(VariablesCollection& variables,
                               const Device src_device, const int src_device_index,
                               const Device dst_device, const int dst_device_index) {
      if (variables.empty())
        return;
      if (src_device == dst_device && src_device_index == dst_device_index)
        return;

      // Move variables back to the CPU device.
      if (src_device != Device::CPU) {
        ScopedDeviceSetter scoped_device_setter(src_device, src_device_index);
        move_variables_to_device(variables, Device::CPU);
      }

      // Move variables to the destination device.
      if (dst_device != Device::CPU) {
        ScopedDeviceSetter scoped_device_setter(dst_device, dst_device_index);
        move_variables_to_device(variables, dst_device);
      }

      synchronize_device(src_device, src_device_index);  // Wait for asynchronous deallocations.
    }

    static StorageView copy_variable(const StorageView& variable,
                                     const Device device, const int device_index) {
      if (variable.is_scalar() || (variable.device() == Device::CPU && device == Device::CPU))
        return variable;

      StorageView copy;

      if (variable.device() != Device::CPU) {
        ScopedDeviceSetter scoped_device_setter(variable.device(), variable.device_index());
        copy = variable.to(Device::CPU);
      }

      if (device != Device::CPU) {
        ScopedDeviceSetter scoped_device_setter(device, device_index);
        if (copy)
          copy = copy.to(device);
        else
          copy = variable.to(device);
      }

      return copy;
    }


    std::unique_ptr<SequenceToSequenceReplica> Model::as_sequence_to_sequence() const {
      throw std::runtime_error("This model cannot be used as a sequence-to-sequence model");
    }

    std::unique_ptr<SequenceGeneratorReplica> Model::as_sequence_generator() const {
      throw std::runtime_error("This model cannot be used as a sequence generator");
    }

    Model::~Model() {
      if (!_variable_index.empty()) {
        _variable_index.clear();
        synchronize_device(_device, _device_index);  // Wait for asynchronous deallocations.
      }
    }

    size_t Model::current_spec_revision() const {
      return 1;
    }

    void Model::set_device(const Device device, const int index) {
      move_variables(_variable_index, _device, _device_index, device, index);
      _device = device;
      _device_index = index;
    }

    void Model::set_compute_type(ComputeType type, Device device, int device_index) {
      if (_device != Device::CPU)
        throw std::runtime_error("set_compute_type expects the variables to be on CPU");

      _saved_compute_type = infer_compute_type();
      _requested_compute_type = type;
      _effective_compute_type = resolve_compute_type(_requested_compute_type,
                                                     _saved_compute_type,
                                                     device,
                                                     device_index);
      _preferred_size_multiple = get_preferred_size_multiple(_effective_compute_type,
                                                             device,
                                                             device_index);

      DataType weight_dtype = DataType::FLOAT32;
      DataType float_dtype = DataType::FLOAT32;
      std::tie(weight_dtype, float_dtype) = compute_type_to_data_type(_effective_compute_type);

      const auto variable_index = _variable_index;
      for (auto& variable_pair : variable_index) {
        const auto& name = variable_pair.first;
        auto& variable = *variable_pair.second;

        // Convert "weight" variables to the expected compute type.
        // Other float variables (e.g. biases) may be converted from or to float16.
        if (is_quantizable(name))
          ensure_dtype(name, variable, weight_dtype);
        else if (is_convertible(variable, name)
                 && is_float_type(variable.dtype())
                 && variable.dtype() != float_dtype)
          variable = variable.to(float_dtype);
      }
    }

    const StorageView* Model::get_variable_if_exists(const std::string& name) const {
      auto it = _variable_index.find(name);
      if (it == _variable_index.end())
        return nullptr;
      return it->second.get();
    }

    const StorageView& Model::get_variable(const std::string& name) const {
      const auto* var = get_variable_if_exists(name);
      if (var == nullptr)
        throw std::out_of_range("variable " + name + " not found");
      return *var;
    }

    std::unordered_map<std::string, StorageView> Model::get_variables() const {
      std::unordered_map<std::string, StorageView> variables;
      variables.reserve(_variable_index.size());
      for (const auto& pair : _variable_index)
        variables.emplace(pair.first, *pair.second);
      return variables;
    }

    bool Model::layer_exists(std::string prefix) const {
      if (!prefix.empty() && prefix.back() != '/')
        prefix += '/';
      for (const auto& pair : _variable_index) {
        const auto& name = pair.first;
        if (starts_with(name, prefix))
          return true;
      }
      return false;
    }

    bool Model::get_flag_with_default(const std::string& name, bool default_value) const {
      return get_attribute_with_default(name, static_cast<int8_t>(default_value));
    }

    void Model::register_variable(std::string name, StorageView variable) {
      _variable_index.emplace(std::move(name), std::make_shared<StorageView>(std::move(variable)));
    }

    void Model::register_variable_alias(std::string alias, const std::string& variable_name) {
      auto it = _variable_index.find(variable_name);
      if (it == _variable_index.end())
        return;
      _variable_index.emplace(std::move(alias), it->second);
    }

    void Model::remove_variable(const std::string& name) {
      _variable_index.erase(name);
    }

    bool Model::is_quantizable(const std::string& variable_name) const {
      return ends_with(variable_name, "weight");
    }

    bool Model::is_linear_weight(const std::string&) const {
      return false;
    }

    bool Model::is_packable(const std::string& variable_name) const {
      return is_linear_weight(variable_name);
    }

    bool Model::is_convertible(const StorageView& variable, const std::string& name) const {
      return !variable.is_scalar() && name.find("_scale") == std::string::npos;
    }

    void Model::ensure_dtype(const std::string& name,
                             StorageView& variable,
                             const DataType target_dtype) {
      const bool is_int8 = variable.dtype() == DataType::INT8;
      const bool is_int16 = variable.dtype() == DataType::INT16;
      const bool is_float32 = variable.dtype() == DataType::FLOAT32;
      const bool is_float16 = variable.dtype() == DataType::FLOAT16;

      const std::string scale_name = name + "_scale";
      const StorageView* saved_scale = nullptr;
      if (is_int8 || is_int16) {
        // Check that the quantization scale of the variable exists.
        saved_scale = get_variable_if_exists(scale_name);
        if (!saved_scale) {
          if (is_int16) {
            // Backward compatibility with int16 models without a saved scale.
            register_variable(scale_name, StorageView(ops::Quantize::global_int16_scale));
            saved_scale = get_variable_if_exists(scale_name);
          } else {
            throw std::runtime_error("variable " + scale_name + " not found");
          }
        }
      }

      if (variable.dtype() == target_dtype)
        return;

      // Use the same quantization logic as in model_spec.py.
      const ops::Quantize quantize_op(/*int16_scale_type=*/ops::Quantize::ScaleType::PER_LAYER,
                                      /*shift_to_uint8=*/false,
                                      /*round_before_cast=*/round_before_cast_in_quantization());
      const ops::Dequantize dequantize_op{};
      StorageView target_variable(target_dtype);

      if (target_dtype == DataType::FLOAT32 || target_dtype == DataType::FLOAT16) {
        if (is_float16) {
          target_variable = variable.to_float();
        } else if (is_float32) {
          target_variable = variable.to_float16();
        } else {
          // Dequantize int8 or int16 back to float32.
          StorageView dequantized;
          dequantize_op(variable, *saved_scale, dequantized);
          remove_variable(scale_name);  // The scale is no longer needed.
          if (target_dtype == DataType::FLOAT16) {
            target_variable = dequantized.to_float16();
          } else {
            target_variable = std::move(dequantized);
          }
        }

      } else if (is_float32 || is_float16) {
        // Quantize float32 to int8 or int16.
        StorageView scale;
        if (is_float16) {
          quantize_op(variable.to_float(), target_variable, scale);
        } else {
          quantize_op(variable, target_variable, scale);
        }
        register_variable(scale_name, std::move(scale));

      } else {
        // Convert int8 -> float32 -> int16 or int16 -> float32 -> int8.
        StorageView tmp_variable;
        StorageView new_scale;
        dequantize_op(variable, *saved_scale, tmp_variable);
        quantize_op(tmp_variable, target_variable, new_scale);
        remove_variable(scale_name);
        register_variable(scale_name, std::move(new_scale));
      }

      variable = std::move(target_variable);
    }

    ComputeType Model::infer_compute_type() const {
      DataType weight_type = DataType::FLOAT32;
      DataType other_type = DataType::FLOAT32;

      for (const auto& variable_pair : _variable_index) {
        const std::string& name = variable_pair.first;
        const StorageView& variable = *variable_pair.second;
        if (is_quantizable(name)) {
          weight_type = variable.dtype();
        } else if (is_convertible(variable, name)) {
          other_type = variable.dtype();
        }
      }

      return data_type_to_compute_type(weight_type, other_type);
    }

    // This method runs some precomputations on linear weights when possible.
    void Model::process_linear_weights() {
      if (_device != Device::CPU)
        return;  // There is currently no processing for non CPU device.

      const bool pack_weights = cpu::pack_gemm_weights(_effective_compute_type);
      const bool transpose = true;
      const float alpha = 1;

      const auto variable_index = _variable_index;
      for (const auto& pair : variable_index) {
        const std::string& name = pair.first;
        if (!is_linear_weight(name))
          continue;

        const StorageView& weight = *pair.second;
        const DataType dtype = weight.dtype();
        const dim_t k = weight.dim(1);
        const dim_t n = weight.dim(0);

        // If the target Gemm implementation prefers the u8s8s32 format, we can shift
        // the input of linear layers to the u8 domain and add a compensation term.
        // This term only depends on the linear weight, so we can compute it once and
        // store it as a model variable.
        if (dtype == DataType::INT8 && cpu::prefer_u8s8s32_gemm()) {
          StorageView compensation = ops::Gemm::compensate_u8_input(weight, transpose, k, n, alpha);
          register_variable(name + "_compensation", std::move(compensation));
        }

        // If requested, linear weights can be packed for the Gemm call.
        if (pack_weights && is_packable(name)) {
          StorageView packed_weight = ops::Gemm::pack_b_input(weight, transpose, k, n, alpha);
          register_variable(name + "_packed", std::move(packed_weight));
          remove_variable(name);  // The original weight is no longer needed.
        }
      }
    }

    static DataType get_dtype_from_item_size(uint8_t item_size) {
      // This is the old (and flawed) logic of resolving the dtype of saved variables.
      switch (item_size) {
      case 4:
        return DataType::FLOAT32;
      case 2:
        return DataType::INT16;
      case 1:
        return DataType::INT8;
      default:
        throw std::runtime_error("unknown data type of width " + std::to_string(item_size));
      }
    }

    static void check_version(const size_t saved_version,
                              const size_t current_version,
                              const std::string& version_type) {
      if (saved_version > current_version)
        throw std::runtime_error("Unsupported model " + version_type
                                 + ". This executable supports models with " + version_type + " v"
                                 + std::to_string(current_version)
                                 + " or below, but the model has " + version_type + " v"
                                 + std::to_string(saved_version)
                                 + ". This usually means that the model was generated by a later "
                                 + "version of CTranslate2. "
                                 + "(Forward compatibility is not guaranteed.)");
    }

    std::shared_ptr<const Model> Model::load(const std::string& path,
                                             Device device,
                                             int device_index,
                                             ComputeType compute_type) {
      ModelFileReader model_reader(path);
      return load(model_reader, device, device_index, compute_type);
    }

    std::shared_ptr<const Model> Model::load(ModelReader& model_reader,
                                             Device device,
                                             int device_index,
                                             ComputeType compute_type) {
      {
        // Log the system configuration the first time a model is loaded.
        static std::once_flag log_once;
        std::call_once(log_once, log_system_config);
      }

      {
        // Check that the device and device index are valid.
        ScopedDeviceSetter(device, device_index);
      }

      std::unique_ptr<std::istream> model_file_ptr = model_reader.get_required_file(binary_file,
                                                                                    /*binary=*/true);
      std::istream& model_file = *model_file_ptr;

      // See the model serialization in python/ctranslate2/specs/model_spec.py.

      // Check the binary version and spec revision.
      const size_t binary_version = consume<uint32_t>(model_file);
      check_version(binary_version, current_binary_version, "binary version");

      std::string spec;
      size_t spec_revision;
      if (binary_version >= 2) {
        spec = consume<std::string>(model_file);
        spec_revision = consume<uint32_t>(model_file);
      } else {
        spec_revision = 1;
      }

      auto model = create_model(spec);
      model->_binary_version = binary_version;
      model->_spec_revision = spec_revision;

      check_version(spec_revision, model->current_spec_revision(), "revision");

      {
        std::unique_ptr<std::istream> config_file_ptr = model_reader.get_file(config_file);
        if (config_file_ptr)
          model->config = nlohmann::json::parse(*config_file_ptr);
      }

      // Load the variables.
      const auto num_variables = consume<uint32_t>(model_file);
      model->_variable_index.reserve(num_variables);
      for (uint32_t i = 0; i < num_variables; ++i) {
        auto name = consume<std::string>(model_file);
        const size_t rank = consume<uint8_t>(model_file);
        const auto* dimensions = consume<uint32_t>(model_file, rank);
        Shape shape(dimensions, dimensions + rank);
        delete [] dimensions;

        DataType dtype;
        dim_t num_bytes = 0;
        if (binary_version >= 4) {
          const auto type_id = consume<uint8_t>(model_file);
          dtype = static_cast<DataType>(type_id);
          num_bytes = consume<uint32_t>(model_file);
        } else {
          const auto item_size = consume<uint8_t>(model_file);
          dtype = get_dtype_from_item_size(item_size);
          num_bytes = consume<uint32_t>(model_file) * item_size;
        }

        StorageView variable(std::move(shape), dtype);
        consume<char>(model_file, num_bytes, static_cast<char*>(variable.buffer()));
        model->register_variable(std::move(name), std::move(variable));
      }

      // Maybe quantize/dequantize/convert the variables to match the requested compute type.
      model->set_compute_type(compute_type, device, device_index);

      // Move variables to the target device.
      model->set_device(device, device_index);

      // Register variable aliases.
      if (binary_version >= 3) {
        const auto num_aliases = consume<uint32_t>(model_file);
        for (uint32_t i = 0; i < num_aliases; ++i) {
          const auto alias = consume<std::string>(model_file);
          const auto variable_name = consume<std::string>(model_file);
          model->register_variable_alias(alias, variable_name);
          // Also alias the quantization scale that could be associated to variable_name.
          model->register_variable_alias(alias + "_scale", variable_name + "_scale");
        }
      }

      // Run additional model initialization.
      const ScopedDeviceSetter scoped_device_setter(device, device_index);
      model->process_linear_weights();
      model->initialize(model_reader);
      return model;
    }

    std::shared_ptr<const Model> Model::copy_to(Device device, int device_index) const {
      auto model = clone();

      // We should consider and keep aliased variables in the new model.
      std::unordered_map<const StorageView*, std::shared_ptr<StorageView>> seen_variables;
      seen_variables.reserve(_variable_index.size());

      for (const auto& pair : _variable_index) {
        const auto& name = pair.first;
        const auto& value = pair.second;

        auto it = seen_variables.find(value.get());

        if (it != seen_variables.end()) {
          model->_variable_index[name] = it->second;
        } else {
          auto copy = std::make_shared<StorageView>(copy_variable(*value, device, device_index));
          model->_variable_index[name] = copy;
          seen_variables.emplace(value.get(), copy);
        }
      }

      model->_device = device;
      model->_device_index = device_index;
      return model;
    }

    bool contains_model(const std::string& path) {
      return bool(ModelFileReader(path).get_file(binary_file));
    }

    ModelLoader::ModelLoader(const std::string& model_path)
      : model_reader(std::make_shared<ModelFileReader>(model_path))
    {
    }

    ModelLoader::ModelLoader(const std::shared_ptr<ModelReader>& model_reader_)
      : model_reader(model_reader_)
    {
    }

    std::vector<std::shared_ptr<const Model>>
    ModelLoader::load() const {
      if (device_indices.empty())
        throw std::invalid_argument("At least one device index should be set");
#ifdef CT2_WITH_CUDA
      if (device == Device::CUDA && !cuda::have_same_compute_capability(device_indices))
        throw std::invalid_argument("Cannot use multiple GPUs with different Compute Capabilities "
                                    "for the same model");
#endif

      std::vector<std::shared_ptr<const Model>> models;
      models.reserve(device_indices.size() * num_replicas_per_device);

      for (const size_t device_index : device_indices) {
        std::shared_ptr<const Model> model;

        if (models.empty())
          model = Model::load(*model_reader, device, device_index, compute_type);
        else
          model = models.back()->copy_to(device, device_index);

        spdlog::info("Loaded model {} on device {}:{}",
                     model_reader->get_model_id(),
                     device_to_str(device),
                     device_index);
        spdlog::info(" - Binary version: {}", model->binary_version());
        spdlog::info(" - Model specification revision: {}", model->spec_revision());
        spdlog::info(" - Selected compute type: {}",
                     compute_type_to_str(model->effective_compute_type()));

        if (model->requested_compute_type() == ComputeType::DEFAULT
            && model->effective_compute_type() != model->saved_compute_type())
          spdlog::warn("The compute type inferred from the saved model is {}, "
                       "but the target device or backend do not support efficient {} computation. "
                       "The model weights have been automatically converted to use "
                       "the {} compute type instead.",
                       compute_type_to_str(model->saved_compute_type()),
                       compute_type_to_str(model->saved_compute_type()),
                       compute_type_to_str(model->effective_compute_type()));

        for (size_t i = 0; i < num_replicas_per_device; ++i)
          models.emplace_back(model);
      }

      return models;
    }

  }
}
