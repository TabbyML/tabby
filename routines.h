#pragma once

#include <mkl.h>

#include "compute.h"
#include "storage_view.h"

std::vector<float*> split_in_depth(const float* input,
                                   size_t batch_size,
                                   size_t depth,
                                   size_t num_splits,
                                   float* output) {
  mkl_somatcopy('R', 'T', batch_size, depth, 1.0 /* alpha */, input, depth, output, batch_size);

  size_t split_size = depth / num_splits;
  for (size_t i = 0; i < num_splits; ++i) {
    float* a = output + (i * split_size * batch_size);
    mkl_simatcopy('R', 'T',
                  split_size, batch_size,
                  1.0 /* alpha */, a,
                  batch_size, split_size);
  }

  std::vector<float*> splits(num_splits);
  for (unsigned i = 0; i < num_splits; ++i) {
    splits[i] = output + (i * batch_size * split_size);
  }
  return splits;
}

static void pad_sequences(const onmt::StorageView& flattened,
                          const onmt::StorageView& lengths,
                          onmt::StorageView& padded) {
  assert(flattened.rank() == 2);
  size_t batch_size = lengths.dim(0);
  size_t max_length = onmt::compute::max(lengths.data<int32_t>(), batch_size);
  size_t depth = flattened.dim(1);
  padded.resize({batch_size, max_length, depth});
  const auto* src = flattened.data<float>();
  auto* dst = padded.data<float>();
  for (size_t i = 0; i < batch_size; ++i) {
    const size_t length = lengths.data<int32_t>()[i];
    size_t count = length * depth;
    onmt::compute::copy(src, dst, count);
    dst += count;
    src += count;
    if (length < max_length) {
      count = (max_length - length) * depth;
      onmt::compute::fill(dst, static_cast<float>(0), count);
      dst += count;
    }
  }
}

static void unpad_sequences(const onmt::StorageView& padded,
                            const onmt::StorageView& lengths,
                            onmt::StorageView& flattened) {
  assert(padded.rank() == 3);
  size_t batch_size = lengths.dim(0);
  size_t max_length = padded.dim(1);
  size_t depth = padded.dim(2);
  size_t total_length = onmt::compute::sum(lengths.data<int32_t>(), batch_size);
  flattened.resize({total_length, depth});
  const auto* src = padded.data<float>();
  auto* dst = flattened.data<float>();
  for (size_t i = 0; i < batch_size; ++i) {
    const size_t length = lengths.data<int32_t>()[i];
    size_t count = depth * length;
    onmt::compute::copy(src, dst, count);
    dst += count;
    src += count + (max_length - length) * depth;
  }
}

static void swap_middle_dims(const onmt::StorageView& x, onmt::StorageView& y) {
  assert(x.rank() == 4);
  size_t d0 = x.dim(0);
  size_t d1 = x.dim(1);
  size_t d2 = x.dim(2);
  size_t d3 = x.dim(3);
  y.resize({d0, d2, d1, d3});
  for (size_t i0 = 0; i0 < d0; ++i0) {
    for (size_t i1 = 0; i1 < d1; ++i1) {
      for (size_t i2 = 0; i2 < d2; ++i2) {
        for (size_t i3 = 0; i3 < d3; ++i3) {
          y.data<float>()[i3 + (i1 * d3) + (i2 * d3 * d1) + (i0 * d3 * d1 * d2)] =
            x.data<float>()[i3 + (i2 * d3) + (i1 * d3 * d2) + (i0 * d3 * d2 * d1)];
        }
      }
    }
  }
}
