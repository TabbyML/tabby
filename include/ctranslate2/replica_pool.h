#pragma once

#include <chrono>

#include "async.h"
#include "models/model.h"
#include "utils.h"

namespace ctranslate2 {

  // Base class to implement a pool of model replicas that can run in parallel.
  class ReplicaPool {
  public:
    virtual ~ReplicaPool() = default;

    ReplicaPool(std::vector<std::unique_ptr<Worker>> workers,
                const long max_queued_batches);

    // Number of batches in the work queue.
    size_t num_queued_batches() const;
    // Number of batches in the work queue or currently processed by a worker.
    size_t num_active_batches() const;
    // Number of parallel replicas.
    size_t num_replicas() const;

  protected:
    template <typename Result>
    void post_examples(const std::vector<Example>& examples,
                       size_t max_batch_size,
                       BatchType batch_type,
                       const BatchJobCreator<Result>& job_creator,
                       const std::shared_ptr<ResultConsumer<Result>>& result_consumer) {
      for (auto& batch : rebatch_input(examples, max_batch_size, batch_type)) {
        auto job = job_creator(std::move(batch));
        job->set_result_consumer(result_consumer);
        _thread_pool->post(std::move(job));
      }
    }

    template <typename Result>
    std::vector<std::future<Result>> post_examples(const std::vector<Example>& examples,
                                                   size_t max_batch_size,
                                                   BatchType batch_type,
                                                   const BatchJobCreator<Result>& job_creator) {
      auto result_consumer = std::make_shared<PromiseSetter<Result>>(examples.size());
      auto futures = result_consumer->get_futures();
      post_examples<Result>(examples, max_batch_size, batch_type, job_creator, result_consumer);
      return futures;
    }

    template <typename ResultWriter, typename Result>
    void consume_batches(BatchReader& batch_reader,
                         ResultWriter& result_writer,
                         const BatchJobCreator<Result>& job_creator,
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
        auto futures = post_examples(examples,
                                     max_batch_size,
                                     batch_type,
                                     job_creator);
        for (auto& future : futures)
          results.emplace(std::move(future));

        pop_results(/*blocking=*/false);
      }

      pop_results(/*blocking=*/true);
    }

    std::unique_ptr<ThreadPool> _thread_pool;
  };


  // Helper functions to create multiple workers for a model.
  template <typename WorkerClass>
  std::vector<std::unique_ptr<Worker>>
  create_workers(size_t num_replicas_per_device,
                 size_t num_threads_per_replica,
                 models::ModelReader& model_reader,
                 const Device device,
                 const std::vector<int>& device_indices,
                 const ComputeType compute_type) {
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
      workers.emplace_back(std::make_unique<WorkerClass>(model, num_threads_per_replica));
    return workers;
  }

  template <typename WorkerClass>
  std::vector<std::unique_ptr<Worker>>
  create_workers(size_t num_replicas_per_device,
                 size_t num_threads_per_replica,
                 const std::string& model_dir,
                 const Device device,
                 const std::vector<int>& device_indices,
                 const ComputeType compute_type) {
    models::ModelFileReader model_reader(model_dir);
    return create_workers<WorkerClass>(num_replicas_per_device,
                                       num_threads_per_replica,
                                       model_reader,
                                       device,
                                       device_indices,
                                       compute_type);
  }

}
