#pragma once

#include "storage_view.h"

namespace ctranslate2 {

  // Base class for sampling from a score distribution.
  class Sampler {
  public:
    virtual ~Sampler() = default;

    // sample_ids and sampled_scores should be on CPU device.
    void operator()(const StorageView& scores,
                    StorageView& sampled_ids,
                    StorageView& sampled_scores,
                    dim_t num_samples = 1) const;
  protected:
    virtual void sample(const StorageView& scores,
                        dim_t num_samples,
                        StorageView& sampled_ids,
                        StorageView& sampled_scores) const = 0;
  };


  class BestSampler : public Sampler {
  protected:
    void sample(const StorageView& scores,
                dim_t num_samples,
                StorageView& sampled_ids,
                StorageView& sampled_scores) const final;
  };


  class RandomSampler : public Sampler {
  public:
    RandomSampler(dim_t from_topk = 0, float topp = 1, float temperature = 1);
  protected:
    void sample(const StorageView& scores,
                dim_t num_samples,
                StorageView& sampled_ids,
                StorageView& sampled_scores) const final;
  private:
    dim_t _from_topk;
    float _topp;
    float _temperature;
  };

}
