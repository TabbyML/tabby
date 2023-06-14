#include "ctranslate2/sampling.h"

#include "ctranslate2/ops/ops.h"

namespace ctranslate2 {

  void Sampler::operator()(const StorageView& scores,
                           StorageView& sampled_ids,
                           StorageView& sampled_scores,
                           dim_t num_samples) const {
    if (sampled_ids.device() != Device::CPU || sampled_scores.device() != Device::CPU)
      throw std::invalid_argument("Sampling outputs should be on the CPU device");
    if (scores.device() == Device::CPU)
      sample(scores, num_samples, sampled_ids, sampled_scores);
    else {
      StorageView sampled_ids_device(DataType::INT32, scores.device());
      StorageView sampled_scores_device(scores.dtype(), scores.device());
      sample(scores, num_samples, sampled_ids_device, sampled_scores_device);
      sampled_ids.copy_from(sampled_ids_device);
      sampled_scores.copy_from(sampled_scores_device);
    }
  }


  void BestSampler::sample(const StorageView& scores,
                           dim_t num_samples,
                           StorageView& sampled_ids,
                           StorageView& sampled_scores) const {
    PROFILE("BestSampler");
    const ops::TopK topk_op(num_samples);
    topk_op(scores, sampled_scores, sampled_ids);
  }


  RandomSampler::RandomSampler(dim_t from_topk, float topp, float temperature)
    : _from_topk(from_topk)
    , _topp(topp)
    , _temperature(temperature) {
  }

  void RandomSampler::sample(const StorageView& scores,
                             dim_t num_samples,
                             StorageView& sampled_ids,
                             StorageView& sampled_scores) const {
    PROFILE("RandomSampler");
    const Device device = scores.device();
    const DataType dtype = scores.dtype();
    const StorageView* final_scores = nullptr;

    // Maybe restrict scores to the best K candidates.
    StorageView top_ids(DataType::INT32, device);
    StorageView top_scores(dtype, device);
    const dim_t total_candidates = scores.dim(-1);
    if (_from_topk > 0 && _from_topk < total_candidates) {
      const ops::TopK topk_op(_from_topk);
      topk_op(scores, top_scores, top_ids);
      final_scores = &top_scores;
    } else if (_from_topk > total_candidates) {
      throw std::invalid_argument("sampling_topk option ("
                                  + std::to_string(_from_topk)
                                  + ") is greater than the vocabulary size ("
                                  + std::to_string(total_candidates)
                                  + ")");
    } else {
      final_scores = &scores;
    }

    // Divide scores by the temperature constant.
    if (_temperature != 1) {
      StorageView scaled_scores(dtype, device);
      ops::Mul()(*final_scores, StorageView(float(1) / _temperature).to(dtype), scaled_scores);
      top_scores = std::move(scaled_scores);
      final_scores = &top_scores;
    }

    if (_topp < 1) {
      StorageView masked_scores(dtype, device);
      const ops::TopPMask mask_op(_topp);
      mask_op(*final_scores, masked_scores);
      top_scores = std::move(masked_scores);
      final_scores = &top_scores;
    }

    // The current Multinomial operator samples with replacement. We can use it when
    // only 1 sample should be returned, otherwise we use the Gumbel-max trick.
    if (num_samples > 1) {
      StorageView log_probs(dtype, device);
      ops::LogSoftMax()(*final_scores, log_probs);
      const ops::GumbelMax gumbel_max_op(num_samples);
      gumbel_max_op(log_probs, sampled_ids);
    } else {
      StorageView probs(dtype, device);
      ops::SoftMax()(*final_scores, probs);
      const ops::Multinomial multinomial_op(num_samples);
      multinomial_op(probs, sampled_ids);
    }

    if (top_ids)  // Return ids relative to the initial distribution.
      ops::Gather(-1, top_ids.rank() - 1)(top_ids, sampled_ids, sampled_ids);
    ops::Gather(-1, scores.rank() - 1)(scores, sampled_ids, sampled_scores);
  }

}
