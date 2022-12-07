#pragma once

#include <algorithm>
#include <limits>

#include "storage_view.h"

namespace ctranslate2 {

  // Helper class to disable tokens in the model output.
  class DisableTokens {
  public:
    DisableTokens(StorageView& logits,
                  const float disable_value = std::numeric_limits<float>::lowest());

    void add(dim_t batch_id, dim_t token_id) {
      const auto flat_index = batch_id * _vocabulary_size + token_id;

      if (_logits_data) {
        // On CPU we directly assign the value.
        _logits_data[flat_index] = _disable_value;

      } else {
        // On GPU we prepare a list of unique index to disable.
        const auto it = std::lower_bound(_flat_indices.begin(), _flat_indices.end(), flat_index);
        if (it == _flat_indices.end() || *it != flat_index)
          _flat_indices.insert(it, flat_index);
      }
    }

    // Disable a token for all batches.
    void add(dim_t token_id) {
      for (dim_t batch_id = 0; batch_id < _batch_size; ++batch_id)
        add(batch_id, token_id);
    }

    void apply();

  private:
    StorageView& _logits;
    float* _logits_data;
    const float _disable_value;
    const dim_t _batch_size;
    const dim_t _vocabulary_size;
    std::vector<int32_t> _flat_indices;
  };

  // Base class for processing the output logits.
  class LogitsProcessor {
  public:
    virtual ~LogitsProcessor() = default;
    virtual void apply(dim_t step,
                       StorageView& logits,
                       DisableTokens& disable_tokens,
                       const StorageView& sequences,
                       const std::vector<dim_t>& batch_offset,
                       const std::vector<std::vector<size_t>>* prefix) = 0;

  protected:
    dim_t get_sample_begin(const dim_t batch_size,
                           const dim_t batch_id,
                           const std::vector<dim_t>& batch_offset,
                           const std::vector<std::vector<size_t>>* prefix) const {
      if (!prefix)
        return 0;
      const auto beam_size = batch_size / batch_offset.size();
      const auto& prefix_tokens = prefix->at(batch_offset[batch_id / beam_size]);
      return prefix_tokens.size();
    }
  };

  // Apply a penalty to the score of previously generated tokens.
  class RepetitionPenalty : public LogitsProcessor {
  public:
    RepetitionPenalty(const float penalty);
    void apply(dim_t step,
               StorageView& logits,
               DisableTokens& disable_tokens,
               const StorageView& sequences,
               const std::vector<dim_t>& batch_offset,
               const std::vector<std::vector<size_t>>* prefix) override;

  private:
    const float _penalty;
  };

  // Prevent repetitions of ngrans with a specific size.
  class NoRepeatNgram : public LogitsProcessor {
  public:
    NoRepeatNgram(const size_t ngram_size);
    void apply(dim_t step,
               StorageView& logits,
               DisableTokens& disable_tokens,
               const StorageView& sequences,
               const std::vector<dim_t>& batch_offset,
               const std::vector<std::vector<size_t>>* prefix) override;

  private:
    const dim_t _ngram_size;
  };

  // Disable the generation of some sequences of tokens.
  class SuppressSequences : public LogitsProcessor {
  public:
    SuppressSequences(std::vector<std::vector<size_t>> sequences);
    void apply(dim_t step,
               StorageView& logits,
               DisableTokens& disable_tokens,
               const StorageView& sequences,
               const std::vector<dim_t>& batch_offset,
               const std::vector<std::vector<size_t>>* prefix) override;

  private:
    std::vector<size_t> _ids;
    std::vector<std::vector<size_t>> _sequences;
  };

  // Disable the generation of some tokens.
  class SuppressTokens : public LogitsProcessor {
  public:
    SuppressTokens(std::vector<size_t> ids);
    void apply(dim_t step,
               StorageView& logits,
               DisableTokens& disable_tokens,
               const StorageView& sequences,
               const std::vector<dim_t>& batch_offset,
               const std::vector<std::vector<size_t>>* prefix) override;

  private:
    const std::vector<size_t> _ids;
  };

  // Disable the generation of some tokens at the first unconstrained decoding step.
  class SuppressTokensBegin : public LogitsProcessor {
  public:
    SuppressTokensBegin(std::vector<size_t> ids);
    void apply(dim_t step,
               StorageView& logits,
               DisableTokens& disable_tokens,
               const StorageView& sequences,
               const std::vector<dim_t>& batch_offset,
               const std::vector<std::vector<size_t>>* prefix) override;

  private:
    const std::vector<size_t> _ids;
  };

}
