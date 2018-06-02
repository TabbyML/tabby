#pragma once

#include <memory>

#include "ops.h"

namespace opennmt {

  class DecoderState {
  public:
    virtual ~DecoderState() = default;

    DecoderState() {
      add("memory", DataType::DT_FLOAT);
      add("memory_lengths", DataType::DT_INT32);
    }

    void reset(const StorageView& memory,
               const StorageView& memory_lengths) {
      get("memory") = memory;
      get("memory_lengths") = memory_lengths;
    }

    void gather(const StorageView& indices) {
      static const ops::Gather gather_op;
      for (auto& pair : _states) {
        gather_op(pair.second, indices);
      }
    }

    StorageView& get(const std::string& name) {
      return _states.at(name);
    }

  protected:
    std::unordered_map<std::string, StorageView> _states;

    void add(const std::string& name, DataType dtype = DataType::DT_FLOAT) {
      _states.emplace(std::piecewise_construct,
                      std::forward_as_tuple(name),
                      std::forward_as_tuple(dtype));
    }
  };

  class Decoder {
  public:
    virtual ~Decoder() = default;

    DecoderState& get_state() {
      return *_state;
    }

    virtual StorageView& logits(size_t step, const StorageView& ids) = 0;

  protected:
    std::unique_ptr<DecoderState> _state;
  };


  void greedy_decoding(Decoder& decoder,
                       StorageView& sample_from,
                       size_t end_token,
                       size_t vocabulary_size,
                       size_t max_steps,
                       std::vector<std::vector<size_t> >& sampled_ids) {
    size_t batch_size = sample_from.dim(0);

    StorageView probs({batch_size, vocabulary_size});
    StorageView alive({batch_size}, DataType::DT_INT32);
    std::vector<bool> finished(batch_size, false);
    std::vector<size_t> batch_offset(batch_size);
    for (size_t i = 0; i < batch_offset.size(); ++i)
      batch_offset[i] = i;
    sampled_ids.resize(batch_size);

    for (size_t step = 0; step < max_steps; ++step) {
      const auto& logits = decoder.logits(step, sample_from);
      ops::SoftMax()(logits, probs);

      std::vector<bool> finished_batch(logits.dim(0), false);
      bool one_finished = false;
      size_t count_alive = 0;
      for (size_t i = 0; i < logits.dim(0); ++i) {
        size_t best = compute::max_element(probs.index<float>({i}), vocabulary_size);
        size_t batch_id = batch_offset[i];
        if (best == end_token) {
          finished[batch_id] = true;
          finished_batch[i] = true;
          one_finished = true;
        } else {
          sample_from.at<int32_t>(i) = best;
          sampled_ids[batch_id].push_back(best);
          ++count_alive;
        }
      }

      if (count_alive == 0)
        break;

      if (one_finished) {
        alive.resize({count_alive});
        size_t write_index = 0;
        size_t read_index = 0;
        for (; read_index < finished_batch.size(); ++read_index) {
          if (!finished_batch[read_index]) {
            batch_offset[write_index] = batch_offset[read_index];
            alive.at<int32_t>(write_index) = read_index;
            ++write_index;
          }
        }
        ops::Gather()(sample_from, alive);
        decoder.get_state().gather(alive);
      }
    }
  }

}
