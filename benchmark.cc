#include <chrono>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>

#include <Eigen/Eigen>
#include <mkl.h>
#include <gemmlowp.h>

#include <emmintrin.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

void Quantize(const float * input, __m128i * output, float quant_mult, int num_rows, int width);
void SSE_MatrixMult(const __m128i * A, const __m128i * B, float * C,
                    float unquant_mult, int num_A_rows, int num_B_rows, int width);

#define BENCHMARK(fun_call, samples)                                    \
  do {                                                                  \
    std::cerr << "benchmarking "#fun_call << std::endl;                 \
    for (int i = 0; i < 10; ++i)                                        \
      fun_call;                                                         \
    std::chrono::high_resolution_clock::time_point t1 =                 \
      std::chrono::high_resolution_clock::now();                        \
    for (int i = 0; i < samples; ++i)                                   \
      fun_call;                                                         \
    std::chrono::high_resolution_clock::time_point t2 =                 \
      std::chrono::high_resolution_clock::now();                        \
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count(); \
    std::cerr << "avg   "                                               \
      << static_cast<double>(duration) / (samples * 1000)               \
              << " ms" << std::endl;                                    \
  } while (false)


std::vector<float> rand_vector(int size) {
  std::vector<float> vec(size);
  for (int i = 0; i < vec.size(); ++i)
    vec[i] = rand();
  return vec;
}

double abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double diff = 0;
  for (int i = 0; i < a.size(); ++i) {
    diff += abs(a[i] - b[i]);
  }
  return diff;
}

static inline
void mean_mkl(const std::vector<float>& input, int batch_size, int depth, std::vector<float>& output) {
  for (int i = 0; i < batch_size; ++i) {
    output[i] = cblas_sasum(depth, input.data() + (i * depth), 1) / depth;
  }
}

static inline
void mean_eigen(const std::vector<float>& input, int batch_size, int depth, std::vector<float>& output) {
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > input_map(input.data(), batch_size, depth);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1> > output_map(output.data(), batch_size, 1);
  output_map.noalias() = input_map.rowwise().mean();
}

void benchmark_mean(int samples = 10000) {
  int batch_size = 64 * 20;
  int depth = 512;

  std::vector<float> input = rand_vector(batch_size * depth);
  std::vector<float> output(batch_size);

  BENCHMARK(mean_mkl(input, batch_size, depth, output), samples);
  BENCHMARK(mean_eigen(input, batch_size, depth, output), samples);
}

static inline
void gemm_eigen(const std::vector<float>& input,
                const std::vector<float>& weight,
                const std::vector<float>& bias,
                int batch_size,
                int input_depth,
                int output_depth,
                std::vector<float>& output) {
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > input_map(input.data(), batch_size, input_depth);
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > weight_map(weight.data(), input_depth, output_depth);
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 1> > bias_map(bias.data(), output_depth, 1);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > output_map(output.data(), batch_size, output_depth);

  output_map.noalias() = input_map * weight_map;
  for (int i = 0; i < batch_size; ++i)
    output_map.row(i).noalias() += bias_map.transpose();
}

static inline
void gemm_bias_before(const std::vector<float>& input,
                      const std::vector<float>& weight,
                      const std::vector<float>& bias,
                      int batch_size,
                      int input_depth,
                      int output_depth,
                      std::vector<float>& output) {
  const int m = batch_size;
  const int n = output_depth;
  const int k = input_depth;
  for (int i = 0; i < m; ++i)
    memcpy(output.data() + (i * n), bias.data(), n * sizeof (float));
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              m, n, k,
              1.0,
              input.data(), k,
              weight.data(), n,
              1.0, output.data(), n);
}

static inline
void gemm_bias_before_parallel(const std::vector<float>& input,
                               const std::vector<float>& weight,
                               const std::vector<float>& bias,
                               int batch_size,
                               int input_depth,
                               int output_depth,
                               std::vector<float>& output) {
  const int m = batch_size;
  const int n = output_depth;
  const int k = input_depth;
  #pragma omp parallel for
  for (int i = 0; i < m; ++i)
    memcpy(output.data() + (i * n), bias.data(), n * sizeof (float));
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              m, n, k,
              1.0,
              input.data(), k,
              weight.data(), n,
              1.0, output.data(), n);
}

static inline
void gemm_bias_before_wt(const std::vector<float>& input,
                         const std::vector<float>& weight_t,
                         const std::vector<float>& bias,
                         int batch_size,
                         int input_depth,
                         int output_depth,
                         std::vector<float>& output) {
  const int m = batch_size;
  const int n = output_depth;
  const int k = input_depth;
  for (int i = 0; i < m; ++i)
    memcpy(output.data() + (i * n), bias.data(), n * sizeof (float));
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              m, n, k,
              1.0,
              input.data(), k,
              weight_t.data(), k,
              1.0, output.data(), n);
}

static inline
void gemm_bias_after(const std::vector<float>& input,
                     const std::vector<float>& weight,
                     const std::vector<float>& bias,
                     int batch_size,
                     int input_depth,
                     int output_depth,
                     std::vector<float>& output) {
  const int m = batch_size;
  const int n = output_depth;
  const int k = input_depth;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              m, n, k,
              1.0,
              input.data(), k,
              weight.data(), n,
              0.0, output.data(), n);
  for (int i = 0; i < m; ++i)
    cblas_saxpy(output_depth, 1.0, bias.data(), 1, output.data() + (i * n), 1);
}

// static inline
// void find_min_max(const std::vector<float>& input, float* min, float* max) {
//   *min = 0;
//   *max = 0;
//   for (const auto f : input) {
//     *min = std::min(*min, f);
//     *max = std::max(*max, f);
//   }
// }

// struct QuantizationParams {
//   float scale;
//   std::uint8_t zero_point;
// };

// // https://github.com/google/gemmlowp/blob/master/doc/quantization_example.cc
// QuantizationParams choose_quantization_params(float min, float max) {
//   min = std::min(min, 0.f);
//   max = std::max(max, 0.f);

//   const float qmin = 0;
//   const float qmax = 255;
//   const double scale = (max - min) / (qmax - qmin);
//   const double initial_zero_point = qmin - min / scale;

//   std::uint8_t nudged_zero_point = 0;
//   if (initial_zero_point < qmin) {
//     nudged_zero_point = qmin;
//   } else if (initial_zero_point > qmax) {
//     nudged_zero_point = qmax;
//   } else {
//     nudged_zero_point =
//         static_cast<std::uint8_t>(std::round(initial_zero_point));
//   }

//   QuantizationParams result;
//   result.scale = scale;
//   result.zero_point = nudged_zero_point;
//   return result;
// }


// void quantize(const QuantizationParams& qparams,
//               const std::vector<float>& src,
//               std::vector<std::uint8_t>& dst) {
//   assert(src.size() == dst.size());
//   for (size_t i = 0; i < src.size(); i++) {
//     const float real_val = src[i];
//     const float transformed_val = qparams.zero_point + real_val / qparams.scale;
//     const float clamped_val = std::max(0.f, std::min(255.f, transformed_val));
//     dst[i] = static_cast<std::uint8_t>(std::round(clamped_val));
//   }
// }

// void dequantize(const QuantizationParams& qparams,
//                 const std::vector<std::uint8_t>& src,
//                 std::vector<float>& dst) {
//   assert(src.size() == dst.size());
//   for (std::size_t i = 0; i < src.size(); i++) {
//     const std::uint8_t quantized_val = src[i];
//     dst[i] = qparams.scale * (quantized_val - qparams.zero_point);
//   }
// }

// static inline
// void gemm_int8(const std::vector<float>& input,
//                const std::vector<std::uint8_t>& weight,
//                const std::vector<float>& bias,
//                int batch_size,
//                int input_depth,
//                int output_depth,
//                std::vector<float>& output) {
//   const float min = 0;
//   const float max = 0;

//   find_min_max(input, &min, &max);
//   auto qparams = choose_quantization_params(min, max);

//   static std::vector<std::uint32_t> qoutput(output.size());
//   static std::vector<std::uint32_t> oc(output.size(), 0);
//   static std::vector<std::uint8_t> qinput(input.size());
//   quantize(qparams, input, qinput);

//   const int m = batch_size;
//   const int n = output_depth;
//   const int k = input_depth;
//   cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasNoTrans, CblasFixOffset, m, n, k, 1.0, qinput.data(), k, 0, weight.data(), n, 0, 0.0, qoutput.data(), n, oc);

//   dequantize(qparams, 

//   for (int i = 0; i < m; ++i)
//     cblas_saxpy(output_depth, 1.0, bias.data(), 1, output.data() + (i * n), 1);
// }

void benchmark_gemm(int samples = 300) {
  int batch_size = 64 * 20;
  int input_depth = 512;
  int output_depth = 2048;

  std::vector<float> weight = rand_vector(input_depth * output_depth);
  std::vector<float> weight_t(weight);
  mkl_simatcopy('R', 'T', input_depth, output_depth, 1.0, weight_t.data(), output_depth, input_depth);
  std::vector<float> bias = rand_vector(output_depth);
  std::vector<float> input = rand_vector(batch_size * input_depth);
  std::vector<float> output(batch_size * output_depth);

  BENCHMARK(gemm_bias_after(input, weight, bias, batch_size, input_depth, output_depth, output),
            samples);
  std::vector<float> o1(output);
  BENCHMARK(gemm_bias_before(input, weight, bias, batch_size, input_depth, output_depth, output),
            samples);
  std::vector<float> o2(output);
  std::cout << "abs diff = " << abs_diff(o1, o2) << std::endl;
  BENCHMARK(gemm_bias_before_wt(input, weight_t, bias, batch_size, input_depth, output_depth, output),
            samples);
  o2 = output;
  std::cout << "abs diff = " << abs_diff(o1, o2) << std::endl;
  BENCHMARK(gemm_bias_before_parallel(input, weight_t, bias, batch_size, input_depth, output_depth, output),
            samples);
}

void gemm_batch(const std::vector<float>& a,
                const std::vector<float>& b,
                MKL_INT batch_size,
                MKL_INT m,
                MKL_INT n,
                MKL_INT k,
                std::vector<float>& c) {
  CBLAS_TRANSPOSE trans_a = CblasNoTrans;
  CBLAS_TRANSPOSE trans_b = CblasNoTrans;
  MKL_INT lda = k;
  MKL_INT ldb = n;
  MKL_INT ldc = n;
  float alpha = 1.0;
  float beta = 0.0;

  std::vector<const float*> a_array(batch_size);
  std::vector<const float*> b_array(batch_size);
  std::vector<float*> c_array(batch_size);
  for (MKL_INT i = 0; i < batch_size; ++i) {
    a_array[i] = a.data() + (i * m * k);
    b_array[i] = b.data() + (i * k * n);
    c_array[i] = c.data() + (i * m * n);
  }

  cblas_sgemm_batch(CblasRowMajor,
                    &trans_a, &trans_b,
                    &m, &n, &k,
                    &alpha, a_array.data(), &lda,
                    b_array.data(), &ldb,
                    &beta, c_array.data(), &ldc,
                    1 /* group_count */, &batch_size);
}

void gemm_batch_loop(const std::vector<float>& a,
                     const std::vector<float>& b,
                     MKL_INT batch_size,
                     MKL_INT m,
                     MKL_INT n,
                     MKL_INT k,
                     std::vector<float>& c) {
  for (MKL_INT i = 0; i < batch_size; ++i) {
    const float* ai = a.data() + (i * m * k);
    const float* bi = b.data() + (i * k * n);
    float* ci = c.data() + (i * m * n);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                1.0, ai, k,
                bi, n,
                0.0, ci, n);
  }
}

void benchmark_gemm_batch(int samples = 3000) {
  int batch_size = 5 * 8;
  int m = 24;
  int n = 27;
  int k = 64;

  std::vector<float> a = rand_vector(batch_size * m * k);
  std::vector<float> b = rand_vector(batch_size * k * n);
  std::vector<float> c(batch_size * m * n);

  BENCHMARK(gemm_batch(a, b, batch_size, m, n, k, c), samples);
  std::vector<float> c1(c);
  BENCHMARK(gemm_batch_loop(a, b, batch_size, m, n, k, c), samples);
  std::vector<float> c2(c);
  std::cout << "abs diff = " << abs_diff(c1, c2) << std::endl;
}

void parallel_memcpy(float* dst, const float* src, int batch_size, int depth) {
  #pragma omp parallel for
  for (int i = 0; i < batch_size; ++i)
    memcpy(dst + (i * depth), src + (i * depth), depth);
}

void memcpy(float* dst, const float* src, int batch_size, int depth) {
  for (int i = 0; i < batch_size; ++i)
    memcpy(dst + (i * depth), src + (i * depth), depth);
}

void benchmark_memcpy(int samples = 10000) {
  int batch_size = 64 * 20;
  int depth = 512;

  std::vector<float> a = rand_vector(batch_size * depth);
  std::vector<float> b(a.size());

  BENCHMARK(memcpy(b.data(), a.data(), batch_size, depth), samples);
  std::vector<float> b1(b);
  BENCHMARK(parallel_memcpy(b.data(), a.data(), batch_size, depth), samples);
  std::vector<float> b2(b);
  std::cout << "abs diff = " << abs_diff(b1, b2) << std::endl;
}


int main() {
  //benchmark_mean();
  //benchmark_gemm();
  //benchmark_gemm_batch();
  benchmark_memcpy();
  return 0;
}
