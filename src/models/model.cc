#include "ctranslate2/models/model.h"

#include <fstream>

#include "ctranslate2/models/transformer.h"
#include "ctranslate2/utils.h"

#include "cpu/backend.h"

namespace ctranslate2 {
  namespace models {

    static const std::string binary_file = "model.bin";

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
      const auto str_length = consume<uint16_t>(in);
      const auto c_str = consume<char>(in, str_length);
      std::string str(c_str);
      delete [] c_str;
      return str;
    }

    static inline void unsupported_compute_type(const std::string& name) {
      throw std::invalid_argument("Requested " + name + " compute type, but the target device "
                                  "or backend do not support efficient " + name + " computation.");
    }

    static DataType compute_type_to_data_type(const ComputeType compute_type,
                                              const DataType data_type,
                                              const Device device,
                                              const bool support_int8,
                                              const bool support_int16,
                                              const bool support_float16) {
      switch (compute_type) {
      case ComputeType::FLOAT: {
        return DataType::FLOAT;
      }
      case ComputeType::FLOAT16: {
        if (!support_float16)
          unsupported_compute_type("float16");
        return DataType::FLOAT16;
      }
      case ComputeType::INT16: {
        if (!support_int16)
          unsupported_compute_type("int16");
        return DataType::INT16;
      }
      case ComputeType::INT8: {
        if (!support_int8)
          unsupported_compute_type("int8");
        return DataType::INT8;
      }
      case ComputeType::DEFAULT: {
        // By default we possibly promote the saved type depending on the hardware support.
        switch (data_type) {
        case DataType::INT16:
          return (support_int16
                  ? DataType::INT16
                  : (device == Device::CPU
                     ? (support_int8 ? DataType::INT8 : DataType::FLOAT)
                     : (support_float16 ? DataType::FLOAT16 : DataType::FLOAT)));
        case DataType::INT8:
          return (support_int8
                  ? DataType::INT8
                  : (support_int16 ? DataType::INT16 : DataType::FLOAT));
        case DataType::FLOAT16:
          return support_float16 ? DataType::FLOAT16 : DataType::FLOAT;
        default:
          return data_type;
        }
      }
      default:
        return data_type;
      }
    }

    static void move_variables_to_device(std::unordered_map<std::string, StorageView>& variables,
                                         const Device device) {
      // Some variables can be shallow copies of others. Those variables should not be
      // moved but updated to point to the new location.
      std::unordered_map<void*, std::vector<StorageView*>> buffer_to_aliases;
      std::vector<StorageView*> variables_to_move;

      buffer_to_aliases.reserve(variables.size());
      variables_to_move.reserve(variables.size());

      // First pass to select the variables to move and map the aliases to the buffer they alias.
      for (auto& pair : variables) {
        StorageView& variable = pair.second;
        if (variable.is_scalar() || variable.device() == device)
          continue;

        if (variable.owns_data()) {
          variables_to_move.push_back(&variable);
        } else {
          buffer_to_aliases[variable.buffer()].push_back(&variable);
        }
      }

      // Second pass to move variables and update the associated aliases.
      for (auto* variable : variables_to_move) {
        void* prev_buffer = variable->buffer();
        *variable = variable->to(device);

        auto it = buffer_to_aliases.find(prev_buffer);
        if (it != buffer_to_aliases.end()) {
          for (StorageView* alias : it->second) {
            StorageView new_alias(alias->dtype(), device);
            new_alias.shallow_copy(*variable);
            swap(*alias, new_alias);
          }
        }
      }
    }

    static void move_variables(std::unordered_map<std::string, StorageView>& variables,
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
    }

    template <typename T>
    static void pack_weight(const StorageView& weight,
                            const bool transpose,
                            const dim_t k,
                            const dim_t n,
                            const float alpha,
                            StorageView& packed_weight) {
      const T* src = weight.data<T>();
      const dim_t pack_bytes = primitives<Device::CPU>::gemm_pack_b(src,
                                                                    transpose,
                                                                    k, n,
                                                                    alpha);

      if (pack_bytes == 0)  // Packed Gemm is not supported.
        return;

      const dim_t pack_size = pack_bytes / sizeof (T);
      const dim_t weight_size = weight.size();

      // We want the packed storage to have the same shape as the original weight
      // so that operators can query its shape, but also have enough space to store
      // the packed data.
      packed_weight.reserve(std::max(weight_size, pack_size));
      packed_weight.resize_as(weight);

      primitives<Device::CPU>::gemm_pack_b(src,
                                           transpose,
                                           k, n,
                                           alpha,
                                           packed_weight.data<T>());
    }


    Model::Model(ModelReader&, size_t spec_revision)
      : _spec_revision(spec_revision) {
    }

    size_t Model::current_spec_revision() const {
      return 1;
    }

    Device Model::device() const {
      return _device;
    }

    int Model::device_index() const
    {
      return _device_index;
    }

    ComputeType Model::compute_type() const {
      return _compute_type;
    }

    void Model::set_device(const Device device, const int index) {
      move_variables(_variable_index, _device, _device_index, device, index);
      _device = device;
      _device_index = index;
    }

    void Model::set_compute_type(ComputeType type) {
      _compute_type = type;
    }

    ScopedDeviceSetter Model::get_scoped_device_setter() const {
      return ScopedDeviceSetter(_device, _device_index);
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

    const std::unordered_map<std::string, StorageView>& Model::get_variables() const {
      return _variable_index;
    }

    bool Model::get_flag_with_default(const std::string& name, bool default_value) const {
      return get_attribute_with_default(name, static_cast<int8_t>(default_value));
    }

    void Model::register_variable(const std::string& name, StorageView& variable) {
      _variable_index.emplace(std::piecewise_construct,
                              std::forward_as_tuple(name),
                              std::forward_as_tuple(std::move(variable)));
    }

    void Model::register_variable_alias(const std::string& alias,
                                        const std::string& variable_name) {
      auto it = _variable_index.find(variable_name);
      if (it == _variable_index.end())
        return;
      StorageView& variable = it->second;
      StorageView view(variable.dtype(), variable.device());
      view.shallow_copy(variable);
      register_variable(alias, view);
    }

    bool Model::is_quantizable(const std::string&) const {
      return false;
    }

    bool Model::is_linear_weight(const std::string&) const {
      return false;
    }

    bool Model::is_packable(const std::string&) const {
      return false;
    }

    void
    Model::ensure_dtype(const std::string& name,
                        StorageView& variable,
                        const DataType target_dtype,
                        std::unordered_map<std::string, StorageView>& variables_to_add,
                        std::vector<std::string>& variables_to_remove) {
      const bool is_int8 = variable.dtype() == DataType::INT8;
      const bool is_int16 = variable.dtype() == DataType::INT16;
      const bool is_float = variable.dtype() == DataType::FLOAT;
      const bool is_float16 = variable.dtype() == DataType::FLOAT16;

      const std::string scale_name = name + "_scale";
      StorageView* saved_scale = nullptr;
      if (is_int8 || is_int16) {
        // Check that the quantization scale of the variable exists.
        auto it = _variable_index.find(scale_name);
        if (it != _variable_index.end()) {
          saved_scale = &it->second;
        } else if (is_int16) {
          // Backward compatibility with int16 models without a saved scale.
          saved_scale = &variables_to_add.emplace(scale_name,
                                                  ops::Quantize::global_int16_scale).first->second;
        } else {
          throw std::runtime_error("variable " + scale_name + " not found");
        }
      }

      if (variable.dtype() == target_dtype)
        return;

      // Use the same quantization logic as in model_spec.py.
      const ops::Quantize quantize_op(/*int16_scale_type=*/ops::Quantize::ScaleType::PER_LAYER);
      const ops::Dequantize dequantize_op{};
      StorageView target_variable(target_dtype);

      if (target_dtype == DataType::FLOAT || target_dtype == DataType::FLOAT16) {
        if (is_float16) {
          target_variable = variable.to_float();
        } else if (is_float) {
          target_variable = variable.to_float16();
        } else {
          // Dequantize int8 or int16 back to float32.
          StorageView dequantized;
          dequantize_op(variable, *saved_scale, dequantized);
          variables_to_remove.emplace_back(scale_name);  // The scale is no longer needed.
          if (target_dtype == DataType::FLOAT16) {
            target_variable = dequantized.to_float16();
          } else {
            target_variable = std::move(dequantized);
          }
        }

      } else if (is_float || is_float16) {
        // Quantize float32 to int8 or int16.
        StorageView scale;
        if (is_float16) {
          quantize_op(variable.to_float(), target_variable, scale);
        } else {
          quantize_op(variable, target_variable, scale);
        }
        variables_to_add.emplace(scale_name, scale);

      } else {
        // Convert int8 -> float32 -> int16 or int16 -> float32 -> int8.
        StorageView tmp_variable;
        dequantize_op(variable, *saved_scale, tmp_variable);
        quantize_op(tmp_variable, target_variable, *saved_scale);
      }

      variable = std::move(target_variable);
    }

    void Model::finalize() {
      auto scoped_device_setter = get_scoped_device_setter();

      const bool support_int8 = mayiuse_int8(_device, _device_index);
      const bool support_int16 = mayiuse_int16(_device, _device_index);
      const bool support_float16 = mayiuse_float16(_device, _device_index);

      std::vector<std::string> variables_to_remove;
      std::unordered_map<std::string, StorageView> variables_to_add;

      for (auto& variable_pair : _variable_index) {
        const auto& name = variable_pair.first;
        auto& variable = variable_pair.second;

        const DataType target_dtype = compute_type_to_data_type(_compute_type,
                                                                variable.dtype(),
                                                                _device,
                                                                support_int8,
                                                                support_int16,
                                                                support_float16);

        // Convert "weight" variables to the expected compute type.
        if (is_quantizable(name)) {
          ensure_dtype(name,
                       variable,
                       target_dtype,
                       variables_to_add,
                       variables_to_remove);
        } else if (!variable.is_scalar() && name.find("_scale") == std::string::npos) {
          // Other parameters may be converted from or to float16 (e.g. bias).
          if (variable.dtype() == DataType::FLOAT && target_dtype == DataType::FLOAT16) {
            variable = variable.to_float16();
          } else if (variable.dtype() == DataType::FLOAT16 && target_dtype == DataType::FLOAT) {
            variable = variable.to_float();
          }
        }
      }

      // Add needed variables.
      for (auto& variable_pair : variables_to_add)
        _variable_index.emplace(std::move(variable_pair));

      // Remove no longer needed variables.
      for (const auto& name : variables_to_remove)
        _variable_index.erase(name);

      // Second pass to move variables on the target device.
      move_variables_to_device(_variable_index, _device);
    }

    // This method runs some precomputations on linear weights when possible.
    void Model::process_linear_weights() {
      if (_device != Device::CPU)
        return;  // There is currently no processing for non CPU device.

      const bool should_pack_weights = cpu::should_pack_gemm_weights();
      const bool transpose = true;
      const float alpha = 1;

      std::vector<std::string> variables_to_remove;
      std::unordered_map<std::string, StorageView> variables_to_add;

      for (const auto& pair : _variable_index) {
        const std::string& name = pair.first;
        if (!is_linear_weight(name))
          continue;

        const StorageView& weight = pair.second;
        const DataType dtype = weight.dtype();
        const dim_t k = weight.dim(1);
        const dim_t n = weight.dim(0);

        // If the target Gemm implementation prefers the u8s8s32 format, we can shift
        // the input of linear layers to the u8 domain and add a compensation term.
        // This term only depends on the linear weight, so we can compute it once and
        // store it as a model variable.
        if (dtype == DataType::INT8 && cpu::prefer_u8s8s32_gemm()) {
          StorageView compensation({n}, DataType::INT32);
          primitives<Device::CPU>::compute_u8_compensation(weight.data<int8_t>(),
                                                           transpose,
                                                           k, n,
                                                           alpha,
                                                           compensation.data<int32_t>());
          variables_to_add.emplace(name + "_compensation", std::move(compensation));
        }

        // If requested, linear weights can be packed for the Gemm call.
        if (should_pack_weights && is_packable(name)) {
          StorageView packed_weight(dtype);

          switch (dtype) {
          case DataType::FLOAT:
            pack_weight<float>(weight, transpose, k, n, alpha, packed_weight);
            break;
          case DataType::INT16:
            pack_weight<int16_t>(weight, transpose, k, n, alpha, packed_weight);
            break;
          case DataType::INT8:
            pack_weight<int8_t>(weight, transpose, k, n, alpha, packed_weight);
            break;
          default:
            break;
          }

          if (!packed_weight.empty()) {
            variables_to_add.emplace(name + "_packed", std::move(packed_weight));
            variables_to_remove.emplace_back(name);  // The original weight is no longer needed.
          }
        }
      }

      for (auto& pair : variables_to_add)
        _variable_index.emplace(std::move(pair));
      for (const auto& name : variables_to_remove)
        _variable_index.erase(name);
    }

    static DataType get_dtype_from_item_size(uint8_t item_size) {
      // This is the old (and flawed) logic of resolving the dtype of saved variables.
      switch (item_size) {
      case 4:
        return DataType::FLOAT;
      case 2:
        return DataType::INT16;
      case 1:
        return DataType::INT8;
      default:
        throw std::runtime_error("unknown data type of width " + std::to_string(item_size));
      }
    }

    static Model* create_model(ModelReader& model_reader,
                               const std::string& spec,
                               size_t spec_revision) {
      Model* model = nullptr;

      // Empty spec name, TransformerBase, and TransformerBig are there for backward
      // compatibility. Now all Transformer variants are saved under TransformerSpec.

      if (spec.empty() || spec == "TransformerBase")
        model = new TransformerModel(model_reader, spec_revision, /*num_heads=*/8);
      else if (spec == "TransformerBig")
        model = new TransformerModel(model_reader, spec_revision, /*num_heads=*/16);
      else if (spec == "TransformerSpec")
        model = new TransformerModel(model_reader, spec_revision);
      else
        throw std::invalid_argument("Unsupported model spec " + spec);

      return model;
    }

    static void check_version(const size_t saved_version,
                              const size_t current_version,
                              const std::string& version_type) {
      if (saved_version > current_version)
        throw std::runtime_error("Unsupported model " + version_type
                                 + ". This executable supports models with " + version_type + " v"
                                 + std::to_string(current_binary_version)
                                 + " or below, but the model has " + version_type + " v"
                                 + std::to_string(saved_version)
                                 + ". This usually means that the model was generated by a later "
                                 + "version of CTranslate2. "
                                 + "(Forward compatibility is not guaranteed.)");
    }

    std::shared_ptr<const Model> Model::load(const std::string& path,
                                             const std::string& device,
                                             int device_index,
                                             const std::string& compute_type) {

      return load(path, str_to_device(device), device_index, str_to_compute_type(compute_type));
    }

    std::shared_ptr<const Model> Model::load(const std::string& path,
                                             Device device,
                                             int device_index,
                                             ComputeType compute_type) {
      ModelFileReader model_reader(path);
      return load(model_reader, device, device_index, compute_type);
    }

    std::shared_ptr<const Model> Model:: load(ModelReader& model_reader,
                                              Device device,
                                              int device_index,
                                              ComputeType compute_type) {
      std::unique_ptr<std::istream> model_file_ptr = model_reader.get_required_file(binary_file,
                                                                                    /*binary=*/true);
      std::istream& model_file = *model_file_ptr;

      // See the model serialization in python/ctranslate2/specs/model_spec.py.
      const auto binary_version = consume<uint32_t>(model_file);
      check_version(binary_version, current_binary_version, "binary version");

      std::string spec;
      size_t spec_revision;
      if (binary_version >= 2) {
        spec = consume<std::string>(model_file);
        spec_revision = consume<uint32_t>(model_file);
      } else {
        spec_revision = 1;
      }

      Model* model = create_model(model_reader, spec, spec_revision);
      model->set_device(device, device_index);
      model->set_compute_type(compute_type);

      check_version(spec_revision, model->current_spec_revision(), "revision");

      const auto num_variables = consume<uint32_t>(model_file);
      model->_variable_index.reserve(num_variables);
      for (uint32_t i = 0; i < num_variables; ++i) {
        const auto name = consume<std::string>(model_file);
        const size_t rank = consume<uint8_t>(model_file);
        const auto* dimensions = consume<uint32_t>(model_file, rank);

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

        StorageView variable({dimensions, dimensions + rank}, dtype);
        consume<char>(model_file, num_bytes, static_cast<char*>(variable.buffer()));
        model->register_variable(name, variable);

        delete [] dimensions;
      }

      model->finalize();

      // Register aliases, which are shallow copies of finalized variables.
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

      model->process_linear_weights();
      return std::shared_ptr<const Model>(model);
    }

    bool contains_model(const std::string& path) {
      return bool(ModelFileReader(path).get_file(binary_file));
    }


    std::unique_ptr<std::istream> ModelReader::get_required_file(const std::string& filename,
                                                                 const bool binary) {
      std::unique_ptr<std::istream> file = get_file(filename, binary);
      if (!file)
        throw std::runtime_error("Unable to open file '" + filename
                                 + "' in model '" + get_model_id() + "'");
      return file;
    }


    ModelFileReader::ModelFileReader(std::string model_dir, std::string path_separator)
      : _model_dir(std::move(model_dir))
      , _path_separator(std::move(path_separator)) {
    }

    std::string ModelFileReader::get_model_id() const {
      return _model_dir;
    }

    std::unique_ptr<std::istream> ModelFileReader::get_file(const std::string& filename,
                                                            const bool binary) {
      const std::string path = _model_dir + _path_separator + filename;
      const std::ios_base::openmode mode = binary ? std::ios_base::binary : std::ios_base::in;
      std::unique_ptr<std::istream> stream(new std::ifstream(path, mode));
      if (!stream || !(*stream))
        return nullptr;
      return stream;
    }

  }
}
