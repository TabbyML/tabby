#include "ctranslate2/sampling.h"

#include "ctranslate2/ops/ops.h"

namespace ctranslate2 {

  void Sampler::operator()(const StorageView& scores,
                           StorageView& sampled_ids,
                           StorageView& sampled_scores,
                           dim_t num_samples) const {
    if (sampled_ids.device() != Device::CPU || sampled_scores.device() != Device::CPU)
      throw std::invalid_argument("Sampling outputs should be on the CPU device");
    sample(scores, num_samples, sampled_ids, sampled_scores);
  }


  void BestSampler::sample(const StorageView& scores,
                           dim_t num_samples,
                           StorageView& sampled_ids,
                           StorageView& sampled_scores) const {
    PROFILE("BestSampler");
    const ops::TopK topk_op(num_samples);
    if (scores.device() == Device::CPU) {
      topk_op(scores, sampled_scores, sampled_ids);
    } else {
      StorageView sampled_ids_device(DataType::DT_INT32, scores.device());
      StorageView sampled_scores_device(scores.dtype(), scores.device());
      topk_op(scores, sampled_scores_device, sampled_ids_device);
      sampled_ids.copy_from(sampled_ids_device);
      sampled_scores.copy_from(sampled_scores_device);
    }
  }


  template <typename T>
  static void select_indices(const StorageView& input,
                             const StorageView& indices,
                             StorageView& output) {
    // Select indices in the depth dimension of input.
    // TODO: optimize this function on CUDA device.
    const T* input_data = input.data<T>();
    StorageView input_host(input.dtype());
    if (input.device() != Device::CPU) {
      input_host.copy_from(input);
      input_data = input_host.data<T>();
    }

    const dim_t depth = input.dim(-1);
    const dim_t batch_size = input.size() / depth;
    const dim_t num_elements = indices.dim(-1);

    output.resize_as(indices);

    for (dim_t i = 0; i < batch_size; ++i) {
      const T* input_row_data = input_data + i * depth;
      const int32_t* indices_row_data = indices.data<int32_t>() + i * num_elements;
      T* output_row_data = output.data<T>() + i * num_elements;

      for (dim_t j = 0; j < num_elements; ++j)
        output_row_data[j] = input_row_data[indices_row_data[j]];
    }
  }

  RandomSampler::RandomSampler(dim_t from_topk, float temperature)
    : _from_topk(from_topk)
    , _temperature(temperature) {
  }

  void RandomSampler::sample(const StorageView& scores,
                             dim_t num_samples,
                             StorageView& sampled_ids,
                             StorageView& sampled_scores) const {
    PROFILE("RandomSampler");
    const Device device = scores.device();
    const StorageView* final_scores = nullptr;

    // Maybe restrict scores to the best K candidates.
    StorageView top_ids(DataType::DT_INT32, device);
    StorageView top_scores(device);
    if (_from_topk > 0) {
      const ops::TopK topk_op(_from_topk);
      topk_op(scores, top_scores, top_ids);
      final_scores = &top_scores;
    } else {
      final_scores = &scores;
    }

    // Divide scores by the temperature constant.
    StorageView scaled_scores(device);
    if (_temperature != 1) {
      ops::Mul()(*final_scores, StorageView(float(1) / _temperature), scaled_scores);
      final_scores = &scaled_scores;
    }

    // Convert scores to probabilities.
    StorageView probs(device);
    ops::SoftMax()(*final_scores, probs);

    // Generate samples.
    // TODO: run Multinomial op on GPU when optimized.
    const ops::Multinomial multinomial_op(num_samples);
    if (device == Device::CPU)
      multinomial_op(probs, sampled_ids);
    else
      multinomial_op(probs.to(Device::CPU), sampled_ids);

    if (_from_topk > 0)  // Return ids relative to the initial distribution.
      select_indices<int32_t>(top_ids, sampled_ids, sampled_ids);
    select_indices<float>(scores, sampled_ids, sampled_scores);
  }

}
