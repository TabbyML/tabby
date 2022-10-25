#pragma once

#include <chrono>
#include <future>

#include "batch_reader.h"
#include "models/model.h"
#include "thread_pool.h"
#include "utils.h"

namespace ctranslate2 {

  template <typename Replica>
  class ReplicaWorker;

  // Base class to implement a pool of model replicas that can run in parallel.
  template <typename Replica>
  class ReplicaPool {
  public:
    virtual ~ReplicaPool() = default;

    ReplicaPool(size_t num_replicas_per_device,
                size_t num_threads_per_replica,
                const std::string& model_dir,
                const Device device,
                const std::vector<int>& device_indices,
                const ComputeType compute_type,
                const long max_queued_batches) {
      models::ModelFileReader model_reader(model_dir);
      initialize_pool(num_replicas_per_device,
                      num_threads_per_replica,
                      model_reader,
                      device,
                      device_indices,
                      compute_type,
                      max_queued_batches);
    }

    ReplicaPool(size_t num_replicas_per_device,
                size_t num_threads_per_replica,
                models::ModelReader& model_reader,
                const Device device,
                const std::vector<int>& device_indices,
                const ComputeType compute_type,
                const long max_queued_batches) {
      initialize_pool(num_replicas_per_device,
                      num_threads_per_replica,
                      model_reader,
                      device,
                      device_indices,
                      compute_type,
                      max_queued_batches);
    }

    // Number of batches in the work queue.
    size_t num_queued_batches() const {
      return _thread_pool->num_queued_jobs();
    }

    // Number of batches in the work queue or currently processed by a worker.
    size_t num_active_batches() const {
      return _thread_pool->num_active_jobs();
    }

    // Number of parallel replicas.
    size_t num_replicas() const {
      return _thread_pool->num_threads();
    }

    // Detaches the models used by each replica for unloading.
    // This method is not thread-safe.
    std::vector<std::shared_ptr<const models::Model>> detach_models() {
      std::vector<std::shared_ptr<const models::Model>> models;
      models.reserve(num_replicas());
      for (size_t i = 0; i < num_replicas(); ++i) {
        auto& worker = static_cast<ReplicaWorker<Replica>&>(_thread_pool->get_worker(i));
        models.emplace_back(worker.detach_model());
      }
      return models;
    }

    // Assigns a model to each replica.
    // This method is not thread-safe.
    void set_models(const std::vector<std::shared_ptr<const models::Model>>& models) {
      if (models.size() != num_replicas())
        throw std::invalid_argument("The number of models does not match the number "
                                    "of parallel replicas");

      for (size_t i = 0; i < num_replicas(); ++i) {
        auto& worker = static_cast<ReplicaWorker<Replica>&>(_thread_pool->get_worker(i));
        worker.set_model(models[i]);
      }
    }

    // Clears the cache of each worker.
    // This method is not thread-safe.
    void clear_cache() const {
      for (size_t i = 0; i < num_replicas(); ++i) {
        auto& worker = static_cast<ReplicaWorker<Replica>&>(_thread_pool->get_worker(i));
        auto* allocator = worker.allocator();
        if (allocator)
          allocator->clear_cache();
      }
    }

  protected:
    template <typename Result, typename Func>
    std::vector<std::future<Result>>
    post_examples(const std::vector<Example>& examples,
                  size_t max_batch_size,
                  BatchType batch_type,
                  const Func& func) {
      std::vector<std::promise<Result>> promises(examples.size());
      std::vector<std::future<Result>> futures;
      futures.reserve(promises.size());
      for (auto& promise : promises)
        futures.emplace_back(promise.get_future());

      post_examples(examples, max_batch_size, batch_type, std::move(promises), func);

      return futures;
    }

    template <typename Result, typename Func>
    void post_examples(const std::vector<Example>& examples,
                       size_t max_batch_size,
                       BatchType batch_type,
                       std::vector<std::promise<Result>> promises,
                       const Func& func) {
      for (auto& batch : rebatch_input(examples, max_batch_size, batch_type)) {
        std::vector<std::promise<Result>> batch_promises;
        batch_promises.reserve(batch.num_examples());
        for (const size_t index : batch.example_index)
          batch_promises.emplace_back(std::move(promises[index]));

        auto job = std::make_unique<BatchJob<Result, Func>>(std::move(batch),
                                                            std::move(batch_promises),
                                                            func);

        _thread_pool->post(std::move(job));
      }
    }

    template <typename Result, typename ResultWriter, typename Func>
    void consume_batches(BatchReader& batch_reader,
                         ResultWriter& result_writer,
                         const Func& func,
                         size_t max_batch_size,
                         size_t read_batch_size,
                         BatchType batch_type) {
      std::queue<std::future<Result>> results;

      auto pop_results = [&results, &result_writer](bool blocking) {
        constexpr std::chrono::seconds zero_sec(0);
        while (!results.empty()
               && (blocking
                   || results.front().wait_for(zero_sec) == std::future_status::ready)) {
          result_writer(results.front().get());
          results.pop();
        }
      };

      if (read_batch_size == 0)
        read_batch_size = (max_batch_size == 1 ? max_batch_size : max_batch_size * 16);

      while (true) {
        auto examples = batch_reader.get_next(read_batch_size, batch_type);
        if (examples.empty())
          break;

        auto futures = post_examples<Result>(examples, max_batch_size, batch_type, func);
        for (auto& future : futures)
          results.emplace(std::move(future));

        pop_results(/*blocking=*/false);
      }

      pop_results(/*blocking=*/true);
    }

  private:
    std::unique_ptr<ThreadPool> _thread_pool;

    void initialize_pool(size_t num_replicas_per_device,
                         size_t num_threads_per_replica,
                         models::ModelReader& model_reader,
                         const Device device,
                         const std::vector<int>& device_indices,
                         const ComputeType compute_type,
                         const long max_queued_batches) {
      // The same number of OpenMP threads should be used for loading and running model.
      if (device == Device::CUDA)
        num_threads_per_replica = 1;
      set_num_threads(num_threads_per_replica);

      const auto models = models::load_replicas(model_reader,
                                                device,
                                                device_indices,
                                                compute_type,
                                                num_replicas_per_device);

      std::vector<std::unique_ptr<Worker>> workers;
      workers.reserve(models.size());
      for (const auto& model : models)
        workers.emplace_back(std::make_unique<ReplicaWorker<Replica>>(model,
                                                                      num_threads_per_replica));

      size_t max_queue_size = std::numeric_limits<size_t>::max();
      if (max_queued_batches == 0)
        max_queue_size = 4 * workers.size();
      else if (max_queued_batches > 0)
        max_queue_size = max_queued_batches;

      _thread_pool = std::make_unique<ThreadPool>(std::move(workers),
                                                  max_queue_size,
                                                  get_core_offset());
    }

    template <typename Result, typename Func>
    class BatchJob : public Job {
    public:
      BatchJob(Batch batch, std::vector<std::promise<Result>> promises, Func func)
        : _batch(std::move(batch))
        , _promises(std::move(promises))
        , _func(std::move(func))
      {
      }

      void run() override {
        std::vector<Result> results;
        std::exception_ptr exception;

        try {
          auto& worker = static_cast<ReplicaWorker<Replica>&>(ThreadPool::get_local_worker());
          auto& replica = worker.replica();
          results = _func(replica, _batch);
        } catch (...) {
          exception = std::current_exception();
        }

        for (size_t i = 0; i < _promises.size(); ++i) {
          if (exception)
            _promises[i].set_exception(exception);
          else
            _promises[i].set_value(std::move(results[i]));
        }
      }

    private:
      const Batch _batch;
      std::vector<std::promise<Result>> _promises;
      const Func _func;
    };

  };


  // Model replica worker.
  template <typename Replica>
  class ReplicaWorker : public Worker {
  public:
    ReplicaWorker(const std::shared_ptr<const models::Model>& model, size_t num_threads)
      : _device(model->device())
      , _device_index(model->device_index())
      , _num_threads(num_threads)
      , _allocator(nullptr)
    {
      set_model(model);
    }

    Replica& replica() {
      if (!_replica)
        throw std::runtime_error("No model replica is available in this thread");
      return *_replica;
    }

    void set_model(const std::shared_ptr<const models::Model>& model) {
      _replica = Replica::create_from_model(*model);
    }

    std::shared_ptr<const models::Model> detach_model() {
      if (!_replica)
        return nullptr;
      auto model = _replica->model();
      _replica.reset();
      return model;
    }

    Allocator* allocator() {
      return _allocator;
    }

  protected:
    void initialize() override {
      // Set the number of OpenMP threads for the current thread.
      set_num_threads(_num_threads);

      // Register the memory allocator used in this thread.
      _allocator = &get_allocator(_device);
    }

    void idle() override {
      // When no new jobs are immediately available, we synchronize the CUDA device
      // so that the CudaAsyncAllocator can release some memory.
      synchronize_device(_device, _device_index);
    }

    void finalize() override {
      _replica.reset();
    }

  private:
    const Device _device;
    const int _device_index;
    const size_t _num_threads;
    Allocator* _allocator;
    std::unique_ptr<Replica> _replica;
  };

}
