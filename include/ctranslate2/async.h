#pragma once

#include <future>

#include "batch_reader.h"
#include "thread_pool.h"

namespace ctranslate2 {

  // Base class for consuming out-of-order results.
  template <typename Result>
  class ResultConsumer {
  public:
    virtual ~ResultConsumer() = default;
    virtual void set_result(size_t index, Result result) = 0;
    virtual void set_exception(size_t index, std::exception_ptr exception) = 0;
  };


  // A result consumer that sets promises.
  template <typename Result>
  class PromiseSetter : public ResultConsumer<Result> {
  public:
    PromiseSetter(size_t num_results)
      : _promises(num_results)
    {
    }

    PromiseSetter(std::vector<std::promise<Result>> promises)
      : _promises(std::move(promises))
    {
    }

    std::vector<std::future<Result>> get_futures() {
      std::vector<std::future<Result>> futures;
      futures.reserve(_promises.size());
      for (auto& promise : _promises)
        futures.emplace_back(promise.get_future());
      return futures;
    }

    void set_result(size_t index, Result result) override {
      _promises[index].set_value(std::move(result));
    }

    void set_exception(size_t index, std::exception_ptr exception) override {
      _promises[index].set_exception(exception);
    }

  private:
    std::vector<std::promise<Result>> _promises;
  };


  // Base class for jobs computing some results from a batch of examples.
  template <typename Result>
  class BatchJob : public Job {
  public:
    BatchJob(Batch batch)
      : _batch(std::move(batch))
    {
    }

    void set_result_consumer(const std::shared_ptr<ResultConsumer<Result>>& consumer) {
      _consumer = consumer;
    }

    void run() override {
      std::vector<Result> results;
      std::exception_ptr exception;

      try {
        results = get_results(_batch);
      } catch (...) {
        exception = std::current_exception();
      }

      if (_consumer) {
        for (size_t i = 0; i < _batch.num_examples(); ++i) {
          const size_t index = (_batch.example_index.empty() ? i : _batch.example_index[i]);
          if (exception)
            _consumer->set_exception(index, exception);
          else
            _consumer->set_result(index, std::move(results[i]));
        }
      }
    }

  protected:
    virtual std::vector<Result> get_results(const Batch& batch) const = 0;

  private:
    const Batch _batch;
    std::shared_ptr<ResultConsumer<Result>> _consumer;
  };


  // Base class to create a job from a batch of examples.
  template <typename Result>
  class BatchJobCreator {
  public:
    virtual std::unique_ptr<BatchJob<Result>> operator()(Batch batch) const = 0;
  };

}
