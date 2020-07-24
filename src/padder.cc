#include "ctranslate2/padder.h"

#include <algorithm>

namespace ctranslate2 {

  Padder::Padder(const StorageView& lengths,
                 const dim_t max_time,
                 const dim_t pad_batch_to_multiple)
    : _batch_size(lengths.size()) {
    const std::vector<int32_t> lengths_vec = lengths.to_vector<int32_t>();
    if (max_time < 0)
      _max_time = *std::max_element(lengths_vec.begin(), lengths_vec.end());
    else
      _max_time = max_time;

    const dim_t max_size = _max_time * _batch_size;
    std::vector<int32_t> padding_to_flat;
    std::vector<int32_t> flat_to_padding;
    padding_to_flat.reserve(max_size);
    flat_to_padding.reserve(max_size);

    dim_t padding_offset = 0;
    dim_t no_padding_offset = 0;

    for (dim_t i = 0; i < _batch_size; ++i) {
      const dim_t length = lengths_vec[i];
      for (dim_t t = 0; t < length; ++t) {
        padding_to_flat.push_back(padding_offset + t);
        flat_to_padding.push_back(no_padding_offset + t);
      }
      for (dim_t t = length; t < _max_time; ++t) {
        flat_to_padding.push_back(no_padding_offset + length - 1);
      }
      padding_offset += _max_time;
      no_padding_offset += length;
    }

    while (padding_to_flat.size() % pad_batch_to_multiple != 0) {
      padding_to_flat.push_back(padding_to_flat.back());
      ++no_padding_offset;
    }

    const Device device = lengths.device();
    _padding_to_flat = StorageView({no_padding_offset}, padding_to_flat, device);
    _flat_to_padding = StorageView({padding_offset}, flat_to_padding, device);
  }

  void Padder::remove_padding(StorageView& x) const {
    Shape shape = x.shape();
    shape[1] *= shape[0];
    shape.erase(shape.begin());
    x.reshape(shape);
    _gather_op(x, _padding_to_flat);
  }

  void Padder::add_padding(StorageView& x) const {
    _gather_op(x, _flat_to_padding);
    Shape shape = x.shape();
    shape[0] /= _batch_size;
    shape.insert(shape.begin(), _batch_size);
    x.reshape(shape);
  }


}
