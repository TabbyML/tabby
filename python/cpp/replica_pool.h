#pragma once

#include <shared_mutex>
#include <ctranslate2/replica_pool.h>

#include "utils.h"

namespace ctranslate2 {
  namespace python {

    inline std::shared_ptr<models::ModelReader>
    create_model_reader(const std::string& model, py::object files) {
      if (files.is_none())
        return std::make_shared<models::ModelFileReader>(model);

      if (!py::isinstance<py::dict>(files))
        throw pybind11::type_error("files argument must be a dictionary mapping file names "
                                   "to the file contents");

      auto reader = std::make_shared<models::ModelMemoryReader>(model);

      for (const auto& pair : files.cast<py::dict>()) {
        auto filename = pair.first;
        auto content = pair.second;

        auto read = py::getattr(content, "read", py::none());
        if (!read.is_none())
          content = read();
        else if (!py::isinstance<py::bytes>(content))
          throw pybind11::type_error("File content must be a file-like or bytes object");

        reader->register_file(filename.cast<std::string>(), content.cast<std::string>());
      }

      return reader;
    }

    template <typename T>
    class ReplicaPoolHelper {
    public:
      ReplicaPoolHelper(const std::string& model_path,
                        const std::string& device,
                        const std::variant<int, std::vector<int>>& device_index,
                        const StringOrMap& compute_type,
                        size_t inter_threads,
                        size_t intra_threads,
                        long max_queued_batches,
                        bool flash_attention,
                        bool tensor_parallel,
                        py::object files)
        : _model_loader(create_model_reader(model_path, files))
        , _device(str_to_device(device))
        , _num_replicas_per_device(inter_threads)
      {
        pybind11::gil_scoped_release nogil;

        _model_loader.device = str_to_device(device);
        _model_loader.device_indices = std::visit(DeviceIndexResolver(), device_index);
        _model_loader.compute_type = std::visit(ComputeTypeResolver(device), compute_type);
        _model_loader.num_replicas_per_device = inter_threads;
        _model_loader.use_flash_attention = flash_attention;
        _model_loader.tensor_parallel = tensor_parallel;

        _pool_config.num_threads_per_replica = intra_threads;
        _pool_config.max_queued_batches = max_queued_batches;

        _pool = std::make_unique<T>(_model_loader, _pool_config);
        _device_index = _model_loader.device_indices;
        _model_is_loaded = true;
      }

      ~ReplicaPoolHelper() {
        pybind11::gil_scoped_release nogil;
        _pool.reset();
      }

      std::string device() const {
        return device_to_str(_model_loader.device);
      }

      const std::vector<int>& device_index() const {
        return _model_loader.device_indices;
      }

      std::string compute_type() const {
        return compute_type_to_str(model()->effective_compute_type());
      }

      bool tensor_parallel() const {
        return _model_loader.tensor_parallel;
      }

      size_t num_replicas() const {
        return _pool->num_replicas();
      }

      size_t num_queued_batches() const {
        return _pool->num_queued_batches();
      }

      size_t num_active_batches() const {
        return _pool->num_active_batches();
      }

      bool model_is_loaded() {
        std::shared_lock lock(_mutex);
        return _model_is_loaded;
      }

      void unload_model(const bool to_cpu) {
        if (to_cpu && _device == Device::CPU)
          return;

        // Do not unload the model if some batches are still being processed.
        if (_pool->num_active_batches() > 0)
          return;

        // If the lock is not acquired immediately it means the model is being used
        // in another thread and we can't unload it at this time.
        std::unique_lock lock(_mutex, std::try_to_lock);
        if (!lock)
          return;

        std::vector<std::shared_ptr<const models::Model>> loaded_models;
        if (_model_is_loaded)
          loaded_models = _pool->detach_models();

        if (to_cpu && _cached_models.empty())
          _cached_models = clone_models(Device::CPU, std::vector<int>(loaded_models.size(), 0), loaded_models);
        else if (!to_cpu)
          _cached_models.clear();
        loaded_models.clear();

        // We clear the CUDA allocator cache to further reduce the memory after unloading the model.
        if (_device == Device::CUDA)
          _pool->clear_cache();

        _model_is_loaded = false;
      }

      void load_model(const bool keep_cache) {
        std::unique_lock lock(_mutex);
        if (_model_is_loaded)
          return;

        std::vector<std::shared_ptr<const models::Model>> loaded_models;
        if (_cached_models.empty())
          loaded_models = _model_loader.load();
        else
          loaded_models = clone_models(_device, _device_index, _cached_models, _num_replicas_per_device);

        _pool->set_models(loaded_models);
        if (!keep_cache)
          _cached_models.clear();
        _model_is_loaded = true;
      }

    protected:
      std::unique_ptr<T> _pool;
      models::ModelLoader _model_loader;
      ReplicaPoolConfig _pool_config;
      const Device _device;
      const size_t _num_replicas_per_device;
      std::vector<int> _device_index;
      std::vector<std::shared_ptr<const models::Model>> _cached_models;
      bool _model_is_loaded;

      // Use a shared mutex to protect the model state (loaded/unloaded).
      // Multiple threads can read the model at the same time, but a single thread can change
      // the model state (e.g. load or unload the model).
      std::shared_mutex _mutex;

      std::vector<std::shared_ptr<const models::Model>> clone_models(Device device,
      const std::vector<int>& device_index,
        std::vector<std::shared_ptr<const models::Model>> cached_models,
        size_t num_models_per_device = 1) {
        std::vector<std::shared_ptr<const models::Model>> copied_models;
        for (size_t i = 0; i < cached_models.size(); ++i) {
          auto& model = const_cast<models::Model&>(*cached_models[i]);
          auto copied_model = model.copy_to(device, device_index[i / num_models_per_device]);
          copied_models.push_back(copied_model);
        }
        return copied_models;
      }

      const std::shared_ptr<const models::Model>& model() const {
        return _pool->get_first_replica().model();
      }

      void assert_model_is_ready() const {
        if (!_model_is_loaded)
          throw std::runtime_error("The model for this translator was unloaded");
      }
    };

  }
}
