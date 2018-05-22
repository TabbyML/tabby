#pragma once

// #include <Eigen/Eigen>
// #include <unsupported/Eigen/CXX11/Tensor>

#include <numeric>
#include <algorithm>

#include <mkl.h>

#define EPSILON 0.000001f

void assert_not_nan(const float* x, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    if (std::isnan(x[i])) {
      throw std::runtime_error("NaN");
    }
  }
}

void concat_in_depth(const std::vector<const float*>& inputs,
                     const std::vector<size_t>& depths,
                     size_t batch_size,
                     float* output) {
  size_t num_inputs = inputs.size();
  size_t total_depth = 0;

  for (size_t i = 0; i < num_inputs; ++i) {
    const size_t depth = depths[i];
    const float* a = inputs[i];
    float* b = output + (total_depth * batch_size);
    mkl_somatcopy('R', 'T', batch_size, depth, 1.0, a, depth, b, batch_size);
    total_depth += depth;
  }

  mkl_simatcopy('R', 'T', total_depth, batch_size, 1.0, output, batch_size, total_depth);
}

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
