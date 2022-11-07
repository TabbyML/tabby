#pragma once

#include <ctranslate2/replica_pool.h>

#include "utils.h"

namespace ctranslate2 {
  namespace python {

    template <typename T>
    class ReplicaPoolHelper {
    public:
      ReplicaPoolHelper(const std::string& model_path,
                        const std::string& device,
                        const std::variant<int, std::vector<int>>& device_index,
                        const StringOrMap& compute_type,
                        size_t inter_threads,
                        size_t intra_threads,
                        long max_queued_batches)
        : _model_loader(model_path)
      {
        _model_loader.device = str_to_device(device);
        _model_loader.device_indices = std::visit(DeviceIndexResolver(), device_index);
        _model_loader.compute_type = std::visit(ComputeTypeResolver(device), compute_type);
        _model_loader.num_replicas_per_device = inter_threads;

        _pool_config.num_threads_per_replica = intra_threads;
        _pool_config.max_queued_batches = max_queued_batches;

        _pool = std::make_unique<T>(_model_loader, _pool_config);
      }

      std::string device() const {
        return device_to_str(_model_loader.device);
      }

      const std::vector<int>& device_index() const {
        return _model_loader.device_indices;
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

    protected:
      std::unique_ptr<T> _pool;
      models::ModelLoader _model_loader;
      ReplicaPoolConfig _pool_config;
    };

  }
}
