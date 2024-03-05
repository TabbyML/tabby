#pragma once

#include <chrono>
#include <future>

#include "batch_reader.h"
#include "models/model.h"
#include "thread_pool.h"
#include "utils.h"

namespace ctranslate2 {

  struct ReplicaPoolConfig {
    size_t num_threads_per_replica = 0;
    long max_queued_batches = 0;
    int cpu_core_offset = -1;
  };

  template <typename Replica>
  class ReplicaWorker;

  // Base class to implement a pool of model replicas that can run in parallel.
  template <typename Replica>
  class ReplicaPool {
  public:
    virtual ~ReplicaPool() = default;

    ReplicaPool(const models::ModelLoader& model_loader,
                const ReplicaPoolConfig& config = {}) {
      initialize_pool(model_loader, config);
    }

    ReplicaPool(const std::string& model_path,
                const Device device,
                const ComputeType compute_type = ComputeType::DEFAULT,
                const std::vector<int>& device_indices = {0},
                const bool tensor_parallel = false,
                const ReplicaPoolConfig& config = {}) {
      models::ModelLoader model_loader(model_path);
      model_loader.device = device;
      model_loader.device_indices = device_indices;
      model_loader.compute_type = compute_type;
      model_loader.tensor_parallel = tensor_parallel;
      initialize_pool(model_loader, config);
    }

    ReplicaPool(const std::shared_ptr<const models::Model>& model,
                const ReplicaPoolConfig& config = {}) {
      initialize_pool({model}, config);
    }

    ReplicaPool(const std::vector<std::shared_ptr<const models::Model>>& replicas,
                const ReplicaPoolConfig& config = {}) {
      initialize_pool(replicas, config);
    }

    // Posts a function and return its result as a future.
    // The function will be run with the first available replica.
    // The function must have the signature: Result(Replica&)
    template <typename Result, typename Func>
    std::future<Result> post(Func func) {
      auto batched_func = [func = std::move(func)](Replica& replica) mutable {
        std::vector<Result> results;
        results.reserve(1);
        results.emplace_back(func(replica));
        return results;
      };

      auto futures = post_batch<Result>(std::move(batched_func), 1);
      return std::move(futures[0]);
    }

    // Posts a function and return one future per result.
    // The function will be run with the first available replica.
    // The function must have the signature: std::vector<Result>(Replica&)
    template <typename Result, typename Func>
    std::vector<std::future<Result>> post_batch(Func func, size_t num_results) {
      std::vector<std::promise<Result>> promises(num_results);
      std::vector<std::future<Result>> futures;
      futures.reserve(promises.size());
      for (auto& promise : promises)
        futures.emplace_back(promise.get_future());

      post_batch(std::move(func), std::move(promises));

      return futures;
    }

    // Same as above, but taking the list of promises directly.
    template <typename Result, typename Func>
    void post_batch(Func func, std::vector<std::promise<Result>> promises) {
      auto wrapped_func = [func = std::move(func)]() mutable {
        return func(get_thread_replica());
      };

      post_func(std::move(wrapped_func), std::move(promises));
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

    const Replica& get_first_replica() const {
      auto& worker = static_cast<ReplicaWorker<Replica>&>(_thread_pool->get_worker(0));
      return worker.replica();
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

        post_batch<Result>(
          [batch = std::move(batch), func](Replica& replica) { return func(replica, batch); },
          std::move(batch_promises));
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

    static Replica& get_thread_replica() {
      auto& worker = static_cast<ReplicaWorker<Replica>&>(ThreadPool::get_local_worker());
      return worker.replica();
    }

    void initialize_pool(const models::ModelLoader& model_loader,
                         const ReplicaPoolConfig& config) {
      // The same number of computation threads should be used for loading and running model.
      set_num_threads(config.num_threads_per_replica);
      initialize_pool(model_loader.load(), config);
    }

    void initialize_pool(const std::vector<std::shared_ptr<const models::Model>>& models,
                         const ReplicaPoolConfig& config) {
      std::vector<std::unique_ptr<Worker>> workers;
      workers.reserve(models.size());
      for (const auto& model : models) {
        workers.emplace_back(std::make_unique<ReplicaWorker<Replica>>(model, config.num_threads_per_replica));
      }

      size_t max_queue_size = std::numeric_limits<size_t>::max();
      if (config.max_queued_batches == 0)
        max_queue_size = 4 * workers.size();
      else if (config.max_queued_batches > 0)
        max_queue_size = config.max_queued_batches;

      _thread_pool = std::make_unique<ThreadPool>(std::move(workers),
                                                  max_queue_size,
                                                  config.cpu_core_offset);
    }

    template <typename Result, typename Func>
    void post_func(Func func, std::vector<std::promise<Result>> promises) {
      _thread_pool->post(std::make_unique<BatchJob<Result, Func>>(std::move(promises),
                                                                  std::move(func)));
    }

    template <typename Result, typename Func>
    class BatchJob : public Job {
    public:
      BatchJob(std::vector<std::promise<Result>> promises, Func func)
        : _promises(std::move(promises))
        , _func(std::move(func))
      {
      }

      void run() override {
        std::vector<Result> results;
        std::exception_ptr exception;

        try {
          results = _func();
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
      std::vector<std::promise<Result>> _promises;
      Func _func;
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
      set_device_index(_device, _device_index);

      // Set the number of computation threads for the current thread.
      set_num_threads(_num_threads);

      // Register the memory allocator used in this thread.
      _allocator = &get_allocator(_device);
    }

    void idle() override {
      // When no new jobs are immediately available, we synchronize the CUDA stream
      // so that the CudaAsyncAllocator can release some memory.
      synchronize_stream(_device);
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
