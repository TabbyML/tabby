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

void array_fill(float* x, float a, size_t size) {
  std::fill_n(x, size, a);
}

void array_copy(const float* x, float* y, size_t size) {
  cblas_scopy(size, x, 1, y, 1);
}

float array_sum(const float* array, size_t size) {
  return std::accumulate(array, array + size, 0.f);
}

float array_mean(const float* array, size_t size) {
  return array_sum(array, size) / size;
}

size_t array_max_element(const float* array, size_t size) {
  return std::max_element(array, array + size) - array;
}

float array_max(const float* array, size_t size) {
  return array[array_max_element(array, size)];
}

void array_add(float a, float* y, size_t size) {
  cblas_saxpy(size, 1.0, &a, 0, y, 1);
}
void array_add(const float* x, float* y, size_t size) {
  cblas_saxpy(size, 1.0 /* a */, x, 1 /* incx */, y, 1 /* incy */);
}

void array_sub(float a, float* y, size_t size) {
  float a_rev = -a;
  array_add(a_rev, y, size);
}

void array_mul(float a, float* y, size_t size) {
  cblas_sscal(size, a, y, 1);
}
void array_mul(const float* x, float* y, size_t size) {
  vsMul(size, y, x, y);
}

void array_pow(const float* x, float *y, float power, size_t size) {
  vsPowx(size, x, power, y);
}

void array_exp(const float* x, float*y, size_t size) {
  vsExp(size, x, y);
}

void sgemm(const float* a,
           const float* b,
           CBLAS_TRANSPOSE trans_a,
           CBLAS_TRANSPOSE trans_b,
           MKL_INT m,
           MKL_INT n,
           MKL_INT k,
           float beta,
           float* c) {
  MKL_INT lda = (trans_a == CblasNoTrans) ? k : m;
  MKL_INT ldb = (trans_b == CblasNoTrans) ? n : k;
  MKL_INT ldc = n;

  cblas_sgemm(CblasRowMajor, trans_a, trans_b,
              m, n, k,
              1.0 /* alpha */, a, lda,
              b, ldb,
              beta, c, ldc);
}

void mat_mul(const float* a,
             const float* b,
             CBLAS_TRANSPOSE trans_a,
             CBLAS_TRANSPOSE trans_b,
             MKL_INT m,
             MKL_INT n,
             MKL_INT k,
             float* c) {
  sgemm(a, b,
        trans_a, trans_b,
        m, n, k,
        0.0 /* beta */, c);
}

void batch_mat_mul(const float* a,
                   const float* b,
                   CBLAS_TRANSPOSE trans_a,
                   CBLAS_TRANSPOSE trans_b,
                   MKL_INT batch_size,
                   MKL_INT m,
                   MKL_INT n,
                   MKL_INT k,
                   float* c) {
  MKL_INT lda = (trans_a == CblasNoTrans) ? k : m;
  MKL_INT ldb = (trans_b == CblasNoTrans) ? n : k;
  MKL_INT ldc = n;
  float alpha = 1.0;
  float beta = 0.0;

  std::vector<const float*> a_array(batch_size);
  std::vector<const float*> b_array(batch_size);
  std::vector<float*> c_array(batch_size);
  for (MKL_INT i = 0; i < batch_size; ++i) {
    a_array[i] = a + (i * m * k);
    b_array[i] = b + (i * k * n);
    c_array[i] = c + (i * m * n);
  }

  cblas_sgemm_batch(CblasRowMajor,
                    &trans_a, &trans_b,
                    &m, &n, &k,
                    &alpha, a_array.data(), &lda,
                    b_array.data(), &ldb,
                    &beta, c_array.data(), &ldc,
                    1 /* group_count */, &batch_size);
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

void linear(const float* input,
            const float* weight,
            const float* bias,
            size_t batch_size,
            size_t input_depth,
            size_t output_depth,
            float* output) {
  MKL_INT m = batch_size;
  MKL_INT n = output_depth;
  MKL_INT k = input_depth;
  float beta = 0.0;

  if (bias != nullptr) {
    beta = 1.0;
    for (MKL_INT i = 0; i < m; ++i)
      array_copy(bias, output + (i * n), n);
  }

  sgemm(input, weight,
        CblasNoTrans, CblasNoTrans,
        m, n, k,
        beta, output);
}

// void test_concat() {
//   std::vector<float> a = {1, 2, 3, 1, 2, 3};
//   std::vector<float> b = {3, 4, 5, 6, 3, 4, 5, 6};
//   std::vector<float> out(a.size() + b.size());
//   //std::vector<std::vector<float> > in({a, b});
//   //concat(engine, in, out, {2, 2}, 1, mkldnn::memory::format::nc);
//   concat_in_depth({a.data(), b.data()}, {3, 4}, 2, out.data());

//   for (auto v : out)
//     std::cout << " " << v;
//   std::cout << std::endl;
// }

// void test_split() {
//   std::vector<float> a = {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
//   std::vector<float> out(a.size());
//   split_in_depth(a.data(), 2, 6, 2, out.data());
//   for (auto v : out)
//     std::cout << " " << v;
//   std::cout << std::endl;
// }
