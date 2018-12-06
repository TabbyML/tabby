#include "ctranslate2/primitives/cpu_mkl.h"

#include <vector>

#include <mkl.h>

#define ALIGNMENT 64

namespace ctranslate2 {

  template<>
  void* primitives<Device::CPU>::alloc_data(size_t size) {
    return mkl_malloc(size, ALIGNMENT);
  }

  template<>
  void primitives<Device::CPU>::free_data(void* data) {
    mkl_free(data);
  }

  template<>
  template<>
  void primitives<Device::CPU>::copy(const float* x, float* y, size_t size) {
    cblas_scopy(size, x, 1 /* incx */, y, 1 /* incy */);
  }

  template<>
  template<>
  void primitives<Device::CPU>::add(const float* a, const float* b, float* c, size_t size) {
    vsAdd(size, a, b, c);
  }

  template<>
  template<>
  void primitives<Device::CPU>::sub(const float* a, const float* b, float* c, size_t size) {
    vsSub(size, a, b, c);
  }

  template<>
  template<>
  void primitives<Device::CPU>::mul(float a, float* y, size_t size) {
    cblas_sscal(size, a, y, 1 /* incx */);
  }

  template<>
  template<>
  void primitives<Device::CPU>::mul(const float* a, const float* b, float* c, size_t size) {
    vsMul(size, a, b, c);
  }

  template<>
  template<>
  void primitives<Device::CPU>::inv(const float* x, float* y, size_t size) {
    vsInv(size, x, y);
  }

  template<>
  template<>
  void primitives<Device::CPU>::pow(const float* x, float *y, float power, size_t size) {
    vsPowx(size, x, power, y);
  }

  template<>
  template<>
  void primitives<Device::CPU>::exp(const float* x, float* y, size_t size) {
    vmsExp(size, x, y, VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
  }

  template<>
  template<>
  void primitives<Device::CPU>::log(const float* x, float* y, size_t size) {
    vmsLn(size, x, y, VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
  }

  template<>
  template<>
  void primitives<Device::CPU>::tanh(const float* x, float* y, size_t size) {
    vsTanh(size, x, y);
  }

  template<>
  template<>
  void primitives<Device::CPU>::transpose_2d(const float* a, const size_t* dims, float* b) {
    auto rows = dims[0];
    auto cols = dims[1];
    mkl_somatcopy('R', 'T', rows, cols, 1.0, a, cols, b, rows);
  }

  template<>
  template<>
  void primitives<Device::CPU>::gemm(const float* a, const float* b,
                                     bool transpose_a, bool transpose_b,
                                     size_t m, size_t n, size_t k,
                                     float alpha, float beta,
                                     float* c) {
    MKL_INT lda = transpose_a ? m : k;
    MKL_INT ldb = transpose_b ? k : n;
    MKL_INT ldc = n;

    MKL_INT m_ = m;
    MKL_INT n_ = n;
    MKL_INT k_ = k;

    CBLAS_TRANSPOSE trans_a = transpose_a ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = transpose_b ? CblasTrans : CblasNoTrans;

    cblas_sgemm(CblasRowMajor,
                trans_a, trans_b,
                m_, n_, k_,
                alpha, a, lda,
                b, ldb,
                beta, c, ldc);
  }

  template<>
  template<>
  void primitives<Device::CPU>::gemm(const int16_t* a, const int16_t* b,
                                     bool transpose_a, bool transpose_b,
                                     size_t m, size_t n, size_t k,
                                     float alpha, float beta,
                                     int32_t* c) {
    MKL_INT lda = transpose_a ? m : k;
    MKL_INT ldb = transpose_b ? k : n;
    MKL_INT ldc = n;

    MKL_INT m_ = m;
    MKL_INT n_ = n;
    MKL_INT k_ = k;

    CBLAS_TRANSPOSE trans_a = transpose_a ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = transpose_b ? CblasTrans : CblasNoTrans;
    CBLAS_OFFSET offsetc = CblasFixOffset;

    MKL_INT16 oa = 0;
    MKL_INT16 ob = 0;
    MKL_INT32 oc = 0;

    cblas_gemm_s16s16s32(CblasRowMajor,
                         trans_a, trans_b,
                         offsetc, m_, n_, k_,
                         alpha,
                         reinterpret_cast<const MKL_INT16*>(a), lda, oa,
                         reinterpret_cast<const MKL_INT16*>(b), ldb, ob,
                         beta,
                         reinterpret_cast<MKL_INT32*>(c), ldc, &oc);
  }

  template<>
  template<>
  void primitives<Device::CPU>::gemm_batch(const float* a, const float* b,
                                           bool transpose_a, bool transpose_b,
                                           size_t batch_size,
                                           size_t m, size_t n, size_t k,
                                           float alpha, float beta,
                                           float* c) {
    MKL_INT lda = transpose_a ? m : k;
    MKL_INT ldb = transpose_b ? k : n;
    MKL_INT ldc = n;

    MKL_INT b_ = batch_size;
    MKL_INT m_ = m;
    MKL_INT n_ = n;
    MKL_INT k_ = k;

    CBLAS_TRANSPOSE trans_a = transpose_a ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = transpose_b ? CblasTrans : CblasNoTrans;

    std::vector<const float*> a_array(batch_size);
    std::vector<const float*> b_array(batch_size);
    std::vector<float*> c_array(batch_size);
    for (MKL_INT i = 0; i < b_; ++i) {
      a_array[i] = a + (i * m_ * k_);
      b_array[i] = b + (i * k_ * n_);
      c_array[i] = c + (i * m_ * n_);
    }

    cblas_sgemm_batch(CblasRowMajor,
                      &trans_a, &trans_b,
                      &m_, &n_, &k_,
                      &alpha, a_array.data(), &lda,
                      b_array.data(), &ldb,
                      &beta, c_array.data(), &ldc,
                      1 /* group_count */, &b_);
  }

}
