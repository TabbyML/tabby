#include "ctranslate2/layers/decoder.h"

#include "ctranslate2/ops/ops.h"

namespace ctranslate2 {
  namespace layers {

    void zero_first_timestep(StorageView& x, dim_t step) {
      if (step == 0) {
        x.zero();
      } else if (step < 0) {
        // TODO: a more direct way to set the first timestep to 0.
        const auto dtype = x.dtype();
        const auto device = x.device();
        StorageView first_step(dtype, device);
        StorageView other_steps(dtype, device);
        ops::Split(1, {1, x.dim(1) - 1})(x, first_step, other_steps);
        first_step.zero();
        ops::Concat(1)({&first_step, &other_steps}, x);
      }
    }


    Decoder::Decoder(Device device)
      : _device(device) {
    }

    void Decoder::gather_state(DecoderState& state, const StorageView& indices) const {
      static const ops::Gather gather_op;

      // When the batch size is unchanged, assume that we are reordering beams.
      bool beam_reordering = indices.size() == batch_size(state);

      for (auto& pair : state) {
        const auto& name = pair.first;
        auto& value = pair.second;
        if (beam_reordering && !should_reorder_state(name))
          continue;
        gather_op(value, indices);
      }
    }

    dim_t Decoder::batch_size(const DecoderState& state) const {
      return state.begin()->second.dim(0);
    }

    bool Decoder::should_reorder_state(const std::string&) const {
      return true;
    }

    Device Decoder::device() const {
      return _device;
    }

  }
}
