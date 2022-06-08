#include "ctranslate2/generator_pool.h"

#include <spdlog/spdlog.h>

namespace ctranslate2 {

  namespace {

    static thread_local models::SequenceGeneratorReplica* local_generator = nullptr;

    class GeneratorWorker : public ReplicaWorker {
    public:
      GeneratorWorker(const std::shared_ptr<const models::Model>& model, size_t num_threads)
        : ReplicaWorker(model->device(), model->device_index(), num_threads)
        , _generator(model->as_sequence_generator())
      {
      }

    protected:
      void initialize() override {
        ReplicaWorker::initialize();
        local_generator = _generator.get();
      }

      void finalize() override {
        _generator.reset();
      }

    private:
      std::unique_ptr<models::SequenceGeneratorReplica> _generator;
    };


    class GenerationJob : public BatchJob<GenerationResult> {
    public:
      GenerationJob(GenerationOptions options, Batch batch)
        : BatchJob(std::move(batch))
        , _options(std::move(options))
      {
      }

    protected:
      std::vector<GenerationResult> get_results(const Batch& batch) const override {
        spdlog::debug("Running batch generation on {} examples", batch.num_examples());
        auto results = local_generator->generate(batch.get_stream(0), _options);
        spdlog::debug("Finished batch generation");
        return results;
      }

    private:
      const GenerationOptions _options;
    };

    class GenerationJobCreator : public BatchJobCreator<GenerationResult> {
    public:
      GenerationJobCreator(GenerationOptions options)
        : _options(std::move(options))
      {
      }

      std::unique_ptr<BatchJob<GenerationResult>> operator()(Batch batch) const {
        return std::make_unique<GenerationJob>(_options, std::move(batch));
      }

    private:
      const GenerationOptions _options;
    };


    class ScoringJob : public BatchJob<ScoringResult> {
    public:
      ScoringJob(ScoringOptions options, Batch batch)
        : BatchJob(std::move(batch))
        , _options(std::move(options))
      {
      }

    protected:
      std::vector<ScoringResult> get_results(const Batch& batch) const override {
        spdlog::debug("Running batch scoring on {} examples", batch.num_examples());
        auto results = local_generator->score(batch.get_stream(0), _options);
        spdlog::debug("Finished batch scoring");
        return results;
      }

    private:
      const ScoringOptions _options;
    };

    class ScoringJobCreator : public BatchJobCreator<ScoringResult> {
    public:
      ScoringJobCreator(ScoringOptions options)
        : _options(std::move(options))
      {
      }

      std::unique_ptr<BatchJob<ScoringResult>> operator()(Batch batch) const {
        return std::make_unique<ScoringJob>(_options, std::move(batch));
      }

    private:
      const ScoringOptions _options;
    };

  }

  GeneratorPool::GeneratorPool(size_t num_generators_per_device,
                               size_t num_threads_per_generator,
                               const std::string& model_dir,
                               const Device device,
                               const std::vector<int>& device_indices,
                               const ComputeType compute_type,
                               const long max_queued_batches)
    : ReplicaPool(create_workers<GeneratorWorker>(num_generators_per_device,
                                                  num_threads_per_generator,
                                                  model_dir,
                                                  device,
                                                  device_indices,
                                                  compute_type),
                  max_queued_batches)
  {
  }

  std::vector<std::future<GenerationResult>>
  GeneratorPool::generate_batch_async(const std::vector<std::vector<std::string>>& start_tokens,
                                      const GenerationOptions& options,
                                      const size_t max_batch_size,
                                      const BatchType batch_type) {
    return post_examples(load_examples({start_tokens}),
                         max_batch_size,
                         batch_type,
                         GenerationJobCreator(options));
  }

  std::vector<std::future<ScoringResult>>
  GeneratorPool::score_batch_async(const std::vector<std::vector<std::string>>& tokens,
                                   const ScoringOptions& options,
                                   const size_t max_batch_size,
                                   const BatchType batch_type) {
    return post_examples(load_examples({tokens}),
                         max_batch_size,
                         batch_type,
                         ScoringJobCreator(options));
  }

}
