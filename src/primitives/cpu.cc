#include "ctranslate2/primitives/primitives.h"

#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>

#ifdef CT2_WITH_MKL
#  include <mkl.h>
#endif

#ifdef CT2_WITH_DNNL
#  include <dnnl.h>
#endif

#include "ctranslate2/utils.h"
#include "cpu/backend.h"
#include "cpu/kernels.h"
#include "cpu/parallel.h"
#include "type_dispatch.h"

#define ALIGNMENT 64

namespace ctranslate2 {

  template<>
  void primitives<Device::CPU>::set_device(int) {
  }

  template<>
  int primitives<Device::CPU>::get_device() {
    return 0;
  }

  template<>
  void* primitives<Device::CPU>::alloc_data(dim_t size, int) {
    return aligned_alloc(size, ALIGNMENT);
  }

  template<>
  void primitives<Device::CPU>::free_data(void* data, int) {
    aligned_free(data);
  }

  template<>
  void primitives<Device::CPU>::clear_cache() {
#ifdef CT2_WITH_MKL
    mkl_free_buffers();
#endif
  }

  template<>
  template <typename T>
  T primitives<Device::CPU>::deref(const T* x, dim_t index) {
    return x[index];
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::fill(T* x, T a, dim_t size) {
    std::fill(x, x + size, a);
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::strided_fill(T* x, T a, dim_t inc_x, dim_t size) {
    for (dim_t i = 0, j = 0; i < size; i++, j += inc_x) {
      x[j] = a;
    }
  }

#ifdef CT2_WITH_MKL
  template<>
  template<>
  void primitives<Device::CPU>::strided_fill(float* x, float a, dim_t inc_x, dim_t size) {
    cblas_scopy(size, &a, /*incx=*/0, x, inc_x);
  }
#endif

  template<>
  template <typename T>
  void primitives<Device::CPU>::copy(const T* x, T* y, dim_t size) {
    std::copy(x, x + size, y);
  }

  template<>
  template <typename T>
  T primitives<Device::CPU>::sum(const T* array, dim_t size) {
    T sum = 0;
    CPU_ISA_DISPATCH((sum = cpu::reduce_sum<ISA>(array, size)));
    return sum;
  }

  template<>
  template <typename T>
  dim_t primitives<Device::CPU>::max_element(const T* array, dim_t size) {
    return std::distance(array, std::max_element(array, array + size));
  }

  template<>
  template <typename T>
  T primitives<Device::CPU>::max(const T* array, dim_t size) {
    T max = 0;
    CPU_ISA_DISPATCH((max = cpu::reduce_max<ISA>(array, size)));
    return max;
  }

  template<>
  template <typename T>
  T primitives<Device::CPU>::amax(const T* array, dim_t size) {
    T max = 0;
    CPU_ISA_DISPATCH((max = cpu::reduce_amax<ISA>(array, size)));
    return max;
  }

  template<>
  template<>
  float primitives<Device::CPU>::amax(const float* x, dim_t size) {
#ifdef CT2_WITH_MKL
    if (cpu::mayiuse_mkl())
      return std::abs(x[cblas_isamax(size, x, /*incx=*/1)]);
#endif
    float max = 0;
    CPU_ISA_DISPATCH((max = cpu::reduce_amax<ISA>(x, size)));
    return max;
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::row_max(const T* x,
                                        const dim_t rows,
                                        const dim_t cols,
                                        T* values,
                                        int32_t* indices) {
    #pragma omp parallel for
    for (dim_t i = 0; i < rows; ++i) {
      const T* row = x + i * cols;
      const T* max = std::max_element(row, row + cols);
      values[i] = *max;
      indices[i] = std::distance(row, max);
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::add(T a, const T* x, T* y, dim_t size) {
    CPU_ISA_DISPATCH((cpu::add<ISA>(a, x, y, size)));
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::add(const T* a, const T* b, T* c, dim_t size) {
    CPU_ISA_DISPATCH((cpu::add<ISA>(a, b, c, size)));
  }

  template<>
  template<>
  void primitives<Device::CPU>::add(const float* a, const float* b, float* c, dim_t size) {
#ifdef CT2_WITH_MKL
    if (cpu::mayiuse_mkl())
      return vsAdd(size, a, b, c);
#endif
    CPU_ISA_DISPATCH((cpu::add<ISA>(a, b, c, size)));
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::add_batch_broadcast(const T* a, const T* b, T* c,
                                                    dim_t a_size, dim_t b_size) {
    const dim_t iter_size = b_size / a_size;
    #pragma omp parallel for
    for (dim_t i = 0; i < iter_size; ++i) {
      const dim_t offset = i * a_size;
      add(a, b + offset, c + offset, a_size);
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::add_depth_broadcast(const T* a, const T* b, T* c,
                                                    dim_t a_size, dim_t b_size) {
    const dim_t iter_size = a_size;
    const dim_t depth = b_size / a_size;
    #pragma omp parallel for
    for (dim_t i = 0; i < iter_size; ++i) {
      const dim_t offset = i * depth;
      add(a[i], b + offset, c + offset, depth);
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::sub(const T* a, const T* b, T* c, dim_t size) {
    CPU_ISA_DISPATCH((cpu::sub<ISA>(a, b, c, size)));
  }

  template<>
  template<>
  void primitives<Device::CPU>::sub(const float* a, const float* b, float* c, dim_t size) {
#ifdef CT2_WITH_MKL
    if (cpu::mayiuse_mkl())
      return vsSub(size, a, b, c);
#endif
    CPU_ISA_DISPATCH((cpu::sub<ISA>(a, b, c, size)));
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::max(T a, const T* x, T* y, dim_t size){
    CPU_ISA_DISPATCH((cpu::max<ISA>(a, x, y, size)));
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::max(const T* a, const T* b, T* c, dim_t size){
    CPU_ISA_DISPATCH((cpu::max<ISA>(a, b, c, size)));
  }

  template<>
  template<>
  void primitives<Device::CPU>::max(const float* a, const float* b, float* c, dim_t size) {
#ifdef CT2_WITH_MKL
    if (cpu::mayiuse_mkl())
      return vsFmax(size, a, b, c);
#endif
    CPU_ISA_DISPATCH((cpu::max<ISA>(a, b, c, size)));
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::min(T a, const T* x, T* y, dim_t size){
    CPU_ISA_DISPATCH((cpu::min<ISA>(a, x, y, size)));
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::min(const T* a, const T* b, T* c, dim_t size){
    CPU_ISA_DISPATCH((cpu::min<ISA>(a, b, c, size)));
  }

  template<>
  template<>
  void primitives<Device::CPU>::min(const float* a, const float* b, float* c, dim_t size) {
#ifdef CT2_WITH_MKL
    if (cpu::mayiuse_mkl())
      return vsFmin(size, a, b, c);
#endif
    CPU_ISA_DISPATCH((cpu::min<ISA>(a, b, c, size)));
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::mul(T a, const T* x, T* y, dim_t size) {
    CPU_ISA_DISPATCH((cpu::mul<ISA>(a, x, y, size)));
  }

  template<>
  template<>
  void primitives<Device::CPU>::mul(float a, const float* x, float* y, dim_t size) {
#ifdef CT2_WITH_MKL
    if (cpu::mayiuse_mkl())
      return cblas_saxpby(size, a, x, 1 /* incx */, 0 /* b */, y, 1 /* incy */);
#endif
    CPU_ISA_DISPATCH((cpu::mul<ISA>(a, x, y, size)));
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::mul(const T* a, const T* b, T* c, dim_t size) {
    CPU_ISA_DISPATCH((cpu::mul<ISA>(a, b, c, size)));
  }

  template<>
  template<>
  void primitives<Device::CPU>::mul(const float* a, const float* b, float* c, dim_t size) {
#ifdef CT2_WITH_MKL
    if (cpu::mayiuse_mkl())
      return vsMul(size, a, b, c);
#endif
    CPU_ISA_DISPATCH((cpu::mul<ISA>(a, b, c, size)));
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::mul_batch_broadcast(const T* a, const T* b, T* c,
                                                    dim_t a_size, dim_t b_size) {
    const dim_t iter_size = b_size / a_size;
    #pragma omp parallel for
    for (dim_t i = 0; i < iter_size; ++i) {
      const dim_t offset = i * a_size;
      mul(a, b + offset, c + offset, a_size);
    }
  }

  template<>
  void primitives<Device::CPU>::relu(const float* x, float* y, dim_t size) {
    cpu::parallel_for(0, size, /*work_size=*/1,
                      [x, y](dim_t begin, dim_t end) {
                        max(float(0), x + begin, y + begin, end - begin);
                      });
  }

  template<>
  void primitives<Device::CPU>::gelu(const float* x, float* y, dim_t size) {
#ifdef CT2_WITH_MKL
    if (cpu::mayiuse_mkl()) {
      const bool inplace = (x == y);
      float* tmp = y;
      if (inplace)
        tmp = static_cast<float*>(alloc_data(size * sizeof (float)));
      vsCdfNorm(size, x, tmp);
      vsMul(size, x, tmp, y);
      if (inplace)
        free_data(tmp);
      return;
    }
#endif
    static const float pi = std::acos(-1.f);
    static const float scale = std::sqrt(2.f / pi);
    cpu::parallel_unary_transform(
      x, y, size, /*work_size=*/14,
      [](float v) {
        return 0.5f * v * (1.f + std::tanh(scale * (v + 0.044715f * std::pow(v, 3.f))));
      });
  }

  template<>
  void primitives<Device::CPU>::exp(const float* x, float* y, dim_t size) {
#ifdef CT2_WITH_MKL
    if (cpu::mayiuse_mkl())
      return vsExp(size, x, y);
#endif
    CPU_ISA_DISPATCH((cpu::exp<ISA>(x, y, size)));
  }

  template<>
  void primitives<Device::CPU>::log(const float* x, float* y, dim_t size) {
#ifdef CT2_WITH_MKL
    if (cpu::mayiuse_mkl())
      return vsLn(size, x, y);
#endif
    CPU_ISA_DISPATCH((cpu::log<ISA>(x, y, size)));
  }

  template<>
  void primitives<Device::CPU>::cos(const float* x, float* y, dim_t size) {
#ifdef CT2_WITH_MKL
    if (cpu::mayiuse_mkl())
      return vsCos(size, x, y);
#endif
    CPU_ISA_DISPATCH((cpu::cos<ISA>(x, y, size)));
  }

  template<>
  void primitives<Device::CPU>::sin(const float* x, float* y, dim_t size) {
#ifdef CT2_WITH_MKL
    if (cpu::mayiuse_mkl())
      return vsSin(size, x, y);
#endif
    CPU_ISA_DISPATCH((cpu::sin<ISA>(x, y, size)));
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::transpose_2d(const T* a, const dim_t* dims, T* b) {
    #pragma omp parallel for
    for (dim_t i0 = 0; i0 < dims[0]; ++i0) {
      for (dim_t i1 = 0; i1 < dims[1]; ++i1) {
        b[i1 * dims[0] + i0] = a[i0 * dims[1] + i1];
      }
    }
  }

#ifdef CT2_WITH_MKL
  template<>
  template<>
  void primitives<Device::CPU>::transpose_2d(const float* a, const dim_t* dims, float* b) {
    const dim_t rows = dims[0];
    const dim_t cols = dims[1];
    mkl_somatcopy('R', 'T', rows, cols, 1.0, a, cols, b, rows);
  }
#endif

  template<>
  template <typename T>
  void primitives<Device::CPU>::transpose_3d(const T* a,
                                             const dim_t* dims,
                                             const dim_t* perm,
                                             T* b) {
    dim_t perm_ind[3];
    for (dim_t i = 0; i < 3; ++i)
      perm_ind[perm[i]] = i;
    const dim_t a_stride[3] = {dims[1] * dims[2], dims[2], 1};
    const dim_t b_stride[3] = {dims[perm[1]] * dims[perm[2]], dims[perm[2]], 1};
    const dim_t perm_b_stride[3] = {b_stride[perm_ind[0]], b_stride[perm_ind[1]],
                                    b_stride[perm_ind[2]]};

    #pragma omp parallel for
    for (dim_t i0 = 0; i0 < dims[0]; ++i0) {
      for (dim_t i1 = 0; i1 < dims[1]; ++i1) {
        for (dim_t i2 = 0; i2 < dims[2]; ++i2) {
          const dim_t b_i = (i0 * perm_b_stride[0] + i1 * perm_b_stride[1] +
                             i2 * perm_b_stride[2]);
          const dim_t a_i = (i0 * a_stride[0] + i1 * a_stride[1] +
                             i2 * a_stride[2]);
          b[b_i] = a[a_i];
        }
      }
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::transpose_4d(const T* a,
                                             const dim_t* dims,
                                             const dim_t* perm,
                                             T* b) {
    if (perm[0] == 0 && perm[1] == 2 && perm[2] == 1 && perm[3] == 3) {
      // Optimize the permutation used in multi-head attention.
      const dim_t r1 = dims[2];
      const dim_t r2 = dims[1];
      const dim_t depth = dims[3];

      #pragma omp parallel for
      for (dim_t i = 0; i < dims[0]; ++i) {
        const dim_t offset = i * r1 * r2;
        for (dim_t j = 0; j < r1 * r2; ++j) {
          const dim_t a_offset = depth * (offset + j);
          const dim_t b_offset = depth * (offset + j / r1 + (j % r1) * r2);
          copy(a + a_offset, b + b_offset, depth);
        }
      }

      return;
    }

    dim_t perm_ind[4];
    for (dim_t i = 0; i < 4; ++i)
      perm_ind[perm[i]] = i;
    const dim_t a_stride[4] = {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
    const dim_t b_stride[4] = {dims[perm[1]] * dims[perm[2]] * dims[perm[3]],
                               dims[perm[2]] * dims[perm[3]], dims[perm[3]], 1};
    const dim_t perm_b_stride[4] = {b_stride[perm_ind[0]], b_stride[perm_ind[1]],
                                    b_stride[perm_ind[2]], b_stride[perm_ind[3]]};

    #pragma omp parallel for
    for (dim_t i0 = 0; i0 < dims[0]; ++i0) {
      for (dim_t i1 = 0; i1 < dims[1]; ++i1) {
        for (dim_t i2 = 0; i2 < dims[2]; ++i2) {
          for (dim_t i3 = 0; i3 < dims[3]; ++i3) {
            const dim_t b_i = (i0 * perm_b_stride[0] + i1 * perm_b_stride[1] +
                               i2 * perm_b_stride[2] + i3 * perm_b_stride[3]);
            const dim_t a_i = (i0 * a_stride[0] + i1 * a_stride[1] +
                               i2 * a_stride[2] + i3 * a_stride[3]);
            b[b_i] = a[a_i];
          }
        }
      }
    }
  }

  static cpu::GemmBackend sgemm_backend = cpu::get_gemm_backend(ComputeType::FLOAT);
  static cpu::GemmBackend gemm_s8_backend = cpu::get_gemm_backend(ComputeType::INT8);
  static cpu::GemmBackend gemm_s16_backend = cpu::get_gemm_backend(ComputeType::INT16);

#ifdef CT2_WITH_MKL
  // m value used to pack the b matrix.
  constexpr MKL_INT mkl_gemm_pack_b_m = 1;
#endif

  template<>
  template<>
  dim_t primitives<Device::CPU>::gemm_pack_b(const float* b,
                                             const bool transpose_b,
                                             const dim_t k,
                                             const dim_t n,
                                             const float alpha,
                                             float* dest) {
#ifdef CT2_WITH_MKL
    if (sgemm_backend == cpu::GemmBackend::MKL) {
      if (!dest)
        return cblas_sgemm_pack_get_size(CblasBMatrix, mkl_gemm_pack_b_m, n, k);
      cblas_sgemm_pack(CblasRowMajor,
                       CblasBMatrix,
                       transpose_b ? CblasTrans : CblasNoTrans,
                       mkl_gemm_pack_b_m, n, k,
                       alpha,
                       b,
                       transpose_b ? k : n,
                       dest);
    }
#endif
    return 0;
  }

  template<>
  template<>
  dim_t primitives<Device::CPU>::gemm_pack_b(const int16_t* b,
                                             const bool transpose_b,
                                             const dim_t k,
                                             const dim_t n,
                                             const float,
                                             int16_t* dest) {
#ifdef CT2_WITH_MKL
    if (gemm_s16_backend == cpu::GemmBackend::MKL) {
      if (!dest)
        return cblas_gemm_s16s16s32_pack_get_size(CblasBMatrix, mkl_gemm_pack_b_m, n, k);
      cblas_gemm_s16s16s32_pack(CblasRowMajor,
                                CblasBMatrix,
                                transpose_b ? CblasTrans : CblasNoTrans,
                                mkl_gemm_pack_b_m, n, k,
                                b,
                                transpose_b ? k : n,
                                dest);
    }
#endif
    return 0;
  }

  template<>
  template<>
  dim_t primitives<Device::CPU>::gemm_pack_b(const int8_t* b,
                                             const bool transpose_b,
                                             const dim_t k,
                                             const dim_t n,
                                             const float,
                                             int8_t* dest) {
#ifdef CT2_WITH_MKL
    if (gemm_s8_backend == cpu::GemmBackend::MKL) {
      if (!dest)
        return cblas_gemm_s8u8s32_pack_get_size(CblasBMatrix, mkl_gemm_pack_b_m, n, k);
      cblas_gemm_s8u8s32_pack(CblasRowMajor,
                              CblasBMatrix,
                              transpose_b ? CblasTrans : CblasNoTrans,
                              mkl_gemm_pack_b_m, n, k,
                              b,
                              transpose_b ? k : n,
                              dest);
    }
#endif
    return 0;
  }

  template<>
  template<>
  void primitives<Device::CPU>::gemm(const float* a, const float* b,
                                     bool a_is_packed, bool b_is_packed,
                                     bool transpose_a, bool transpose_b,
                                     dim_t m, dim_t n, dim_t k,
                                     float alpha, float beta,
                                     float* c,
                                     const float*) {
    const dim_t lda = transpose_a ? m : k;
    const dim_t ldb = transpose_b ? k : n;
    const dim_t ldc = n;

    switch (sgemm_backend) {

#ifdef CT2_WITH_MKL
    case cpu::GemmBackend::MKL: {
      CBLAS_TRANSPOSE trans_a = transpose_a ? CblasTrans : CblasNoTrans;
      CBLAS_TRANSPOSE trans_b = transpose_b ? CblasTrans : CblasNoTrans;

      if (a_is_packed || b_is_packed) {
        cblas_sgemm_compute(CblasRowMajor,
                            a_is_packed ? (MKL_INT)CblasPacked : (MKL_INT)trans_a,
                            b_is_packed ? (MKL_INT)CblasPacked : (MKL_INT)trans_b,
                            m, n, k,
                            a, lda,
                            b, ldb,
                            beta, c, ldc);
      } else {
        cblas_sgemm(CblasRowMajor,
                    trans_a, trans_b,
                    m, n, k,
                    alpha,
                    a, lda,
                    b, ldb,
                    beta,
                    c, ldc);
      }
      break;
    }
#endif

#ifdef CT2_WITH_DNNL
    case cpu::GemmBackend::DNNL: {
      dnnl_sgemm(transpose_a ? 'T' : 'N',
                 transpose_b ? 'T' : 'N',
                 m, n, k,
                 alpha,
                 a, lda,
                 b, ldb,
                 beta,
                 c, ldc);
      break;
    }
#endif

    default:
      throw std::runtime_error("No SGEMM backend on CPU");
    }
  }

  template<>
  template<>
  void primitives<Device::CPU>::gemm(const int16_t* a, const int16_t* b,
                                     bool a_is_packed, bool b_is_packed,
                                     bool transpose_a, bool transpose_b,
                                     dim_t m, dim_t n, dim_t k,
                                     float alpha, float beta,
                                     int32_t* c,
                                     const int32_t*) {
    switch (gemm_s16_backend) {

#ifdef CT2_WITH_MKL
    case cpu::GemmBackend::MKL: {
      MKL_INT lda = transpose_a ? m : k;
      MKL_INT ldb = transpose_b ? k : n;
      MKL_INT ldc = n;

      CBLAS_TRANSPOSE trans_a = transpose_a ? CblasTrans : CblasNoTrans;
      CBLAS_TRANSPOSE trans_b = transpose_b ? CblasTrans : CblasNoTrans;
      CBLAS_OFFSET offsetc = CblasFixOffset;

      MKL_INT16 oa = 0;
      MKL_INT16 ob = 0;
      MKL_INT32 oc = 0;

      if (a_is_packed || b_is_packed) {
        cblas_gemm_s16s16s32_compute(CblasRowMajor,
                                     a_is_packed ? (MKL_INT)CblasPacked : (MKL_INT)trans_a,
                                     b_is_packed ? (MKL_INT)CblasPacked : (MKL_INT)trans_b,
                                     offsetc, m, n, k,
                                     alpha,
                                     a, lda, oa,
                                     b, ldb, ob,
                                     beta,
                                     c, ldc, &oc);
      } else {
        cblas_gemm_s16s16s32(CblasRowMajor,
                             trans_a, trans_b,
                             offsetc, m, n, k,
                             alpha,
                             a, lda, oa,
                             b, ldb, ob,
                             beta,
                             c, ldc, &oc);

      }
      break;
    }
#endif

    default:
      throw std::runtime_error("No INT16 GEMM backend on CPU");
    }
  }

#ifdef CT2_WITH_MKL
  static void shift_to_u8(const int8_t* x, uint8_t* ux, dim_t size) {
    cpu::unary_transform(x, ux, size, [](int8_t v) { return static_cast<uint8_t>(v + 128); });
  }
#endif

  template<>
  void primitives<Device::CPU>::compute_u8_compensation(const int8_t* b,
                                                        bool transpose_b,
                                                        dim_t k,
                                                        dim_t n,
                                                        float alpha,
                                                        int32_t* compensation) {
    #pragma omp parallel for
    for (dim_t i = 0; i < n; ++i) {
      int32_t val = 0;

      if (transpose_b) {
        const int8_t* row = b + i * k;
        val = std::accumulate(row, row + k, static_cast<int32_t>(0));
      } else {
        for (dim_t j = 0; j < k; ++j) {
          val += b[j * n + i];
        }
      }

      if (alpha != 1) {
        val = static_cast<int32_t>(static_cast<float>(val) * alpha * -128.0);
      } else {
        val *= -128;
      }

      compensation[i] = val;
    }
  }


  template<>
  template<>
  void primitives<Device::CPU>::gemm(const int8_t* a, const int8_t* b,
                                     bool a_is_packed, bool b_is_packed,
                                     bool transpose_a, bool transpose_b,
                                     dim_t m, dim_t n, dim_t k,
                                     float alpha, float beta,
                                     int32_t* c,
                                     const int32_t* a_shift_compensation) {
    const dim_t lda = transpose_a ? m : k;
    const dim_t ldb = transpose_b ? k : n;
    const dim_t ldc = n;

    switch (gemm_s8_backend) {

#ifdef CT2_WITH_MKL
    case cpu::GemmBackend::MKL: {
      // We are implementing s8s8s32 GEMM with cblas_gemm_s8u8s32. In row major mode,
      // it expects a to be unsigned and b to be signed. So we need to shift a to the
      // uint8 domain and add a compensation term. For more details, see
      // https://intel.github.io/mkl-dnn/dev_guide_int8_computations.html

      const bool use_packed_api = a_is_packed || b_is_packed;
      const uint8_t* ua = nullptr;
      uint8_t* tmp_ua = nullptr;
      int32_t* tmp_a_shift_compensation = nullptr;

      if (a_shift_compensation) {
        // If the compensation term is passed as argument, we assume a is already shifted.
        ua = reinterpret_cast<const uint8_t*>(a);
      } else if (use_packed_api) {
        throw std::invalid_argument("Packed cblas_gemm_s8u8s32 requires the uint8 shift "
                                    "compensation term to be passed as argument");
      } else {
        const dim_t a_size = m * k;
        tmp_ua = static_cast<uint8_t*>(alloc_data(a_size));
        shift_to_u8(a, tmp_ua, a_size);
        ua = tmp_ua;

        tmp_a_shift_compensation = static_cast<int32_t*>(alloc_data(n * sizeof (int32_t)));
        compute_u8_compensation(b, transpose_b, k, n, alpha, tmp_a_shift_compensation);
        a_shift_compensation = tmp_a_shift_compensation;
      }

      const CBLAS_TRANSPOSE trans_a = transpose_a ? CblasTrans : CblasNoTrans;
      const CBLAS_TRANSPOSE trans_b = transpose_b ? CblasTrans : CblasNoTrans;

      if (use_packed_api) {
        cblas_gemm_s8u8s32_compute(CblasRowMajor,
                                   a_is_packed ? (MKL_INT)CblasPacked : (MKL_INT)trans_a,
                                   b_is_packed ? (MKL_INT)CblasPacked : (MKL_INT)trans_b,
                                   CblasRowOffset,
                                   m, n, k,
                                   alpha,
                                   ua, lda, 0,
                                   b, ldb, 0,
                                   beta,
                                   c, ldc, a_shift_compensation);
      } else {
        cblas_gemm_s8u8s32(CblasRowMajor,
                           trans_a, trans_b,
                           CblasRowOffset,
                           m, n, k,
                           alpha,
                           ua, lda, 0,
                           b, ldb, 0,
                           beta,
                           c, ldc, a_shift_compensation);
      }

      if (tmp_ua)
        free_data(tmp_ua);
      if (tmp_a_shift_compensation)
        free_data(tmp_a_shift_compensation);
      break;
    }
#endif

#ifdef CT2_WITH_DNNL
    case cpu::GemmBackend::DNNL: {
      const char transa = transpose_a ? 'T' : 'N';
      const char transb = transpose_b ? 'T' : 'N';

      if (a_shift_compensation) {
        // If the compensation term is passed as argument, we assume a is already shifted.
        dnnl_gemm_u8s8s32(transa, transb,
                          'R',
                          m, n, k,
                          alpha,
                          reinterpret_cast<const uint8_t*>(a), lda, 0,
                          b, ldb, 0,
                          beta,
                          c, ldc, a_shift_compensation);
      } else {
        const int32_t co = 0;
        dnnl_gemm_s8s8s32(transa, transb,
                          'F',
                          m, n, k,
                          alpha,
                          a, lda, 0,
                          b, ldb, 0,
                          beta,
                          c, ldc, &co);
      }
      break;
    }
#endif

    default:
      throw std::runtime_error("No INT8 GEMM backend for CPU");
    }
  }

  template<>
  template<>
  void primitives<Device::CPU>::gemm_batch(const float* a, const float* b,
                                           bool transpose_a, bool transpose_b,
                                           dim_t batch_size,
                                           dim_t m, dim_t n, dim_t k,
                                           float alpha, float beta,
                                           float* c) {
    switch (sgemm_backend) {

#ifdef CT2_WITH_MKL
    case cpu::GemmBackend::MKL: {
      MKL_INT lda = transpose_a ? m : k;
      MKL_INT ldb = transpose_b ? k : n;
      MKL_INT ldc = n;

      MKL_INT b_ = batch_size;
      MKL_INT m_ = m;
      MKL_INT n_ = n;
      MKL_INT k_ = k;

      CBLAS_TRANSPOSE trans_a = transpose_a ? CblasTrans : CblasNoTrans;
      CBLAS_TRANSPOSE trans_b = transpose_b ? CblasTrans : CblasNoTrans;

      auto ptr_array = static_cast<float**>(alloc_data(3 * batch_size * sizeof (float*)));
      auto a_array = const_cast<const float**>(ptr_array);
      auto b_array = const_cast<const float**>(ptr_array + batch_size);
      auto c_array = ptr_array + 2 * batch_size;
      for (MKL_INT i = 0; i < b_; ++i) {
        a_array[i] = a + (i * m_ * k_);
        b_array[i] = b + (i * k_ * n_);
        c_array[i] = c + (i * m_ * n_);
      }

      cblas_sgemm_batch(CblasRowMajor,
                        &trans_a, &trans_b,
                        &m_, &n_, &k_,
                        &alpha, a_array, &lda,
                        b_array, &ldb,
                        &beta, c_array, &ldc,
                        1 /* group_count */, &b_);

      free_data(ptr_array);
      break;
    }
#endif

    default: {
      #pragma omp parallel for
      for (dim_t i = 0; i < batch_size; ++i) {
        const float* a_i = a + (i * m * k);
        const float* b_i = b + (i * k * n);
        float* c_i = c + (i * m * n);

        gemm(a_i, b_i,
             /*a_is_packed=*/false, /*b_is_packed=*/false,
             transpose_a, transpose_b,
             m, n, k,
             alpha, beta,
             c_i);
      }
      break;
    }

    }
  }


#define DECLARE_IMPL(T)                                                 \
  template T                                                            \
  primitives<Device::CPU>::deref(const T* x, dim_t index);              \
  template void                                                         \
  primitives<Device::CPU>::fill(T* x, T a, dim_t size);                 \
  template void                                                         \
  primitives<Device::CPU>::strided_fill(T* x, T a, dim_t inc_x, dim_t size); \
  template void                                                         \
  primitives<Device::CPU>::copy(const T* x, T* y, dim_t size);          \
  template T                                                            \
  primitives<Device::CPU>::sum(const T* array, dim_t size);             \
  template dim_t                                                        \
  primitives<Device::CPU>::max_element(const T* array, dim_t size);     \
  template T                                                            \
  primitives<Device::CPU>::max(const T* array, dim_t size);             \
  template T                                                            \
  primitives<Device::CPU>::amax(const T* array, dim_t size);            \
  template void                                                         \
  primitives<Device::CPU>::row_max(const T* x,                          \
                                   const dim_t rows,                    \
                                   const dim_t cols,                    \
                                   T* values,                           \
                                   int32_t* indices);                   \
  template void                                                         \
  primitives<Device::CPU>::add(T a, const T* x, T* y, dim_t size);      \
  template void                                                         \
  primitives<Device::CPU>::add(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CPU>::add_batch_broadcast(const T* a, const T* b, T* c, \
                                               dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::CPU>::add_depth_broadcast(const T* a, const T* b, T* c, \
                                               dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::CPU>::sub(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CPU>::min(T a, const T* x, T* y, dim_t size);      \
  template void                                                         \
  primitives<Device::CPU>::min(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CPU>::max(T a, const T* x, T* y, dim_t size);     \
  template void                                                         \
  primitives<Device::CPU>::max(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CPU>::mul(T a, const T* x, T* y, dim_t size);      \
  template void                                                         \
  primitives<Device::CPU>::mul(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CPU>::mul_batch_broadcast(const T* a, const T* b, T* c, \
                                               dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::CPU>::transpose_2d(const T* a,                     \
                                        const dim_t* dims,              \
                                        T* b);                          \
  template void                                                         \
  primitives<Device::CPU>::transpose_3d(const T* a,                     \
                                        const dim_t* dims,              \
                                        const dim_t* perm,              \
                                        T* b);                          \
  template void                                                         \
  primitives<Device::CPU>::transpose_4d(const T* a,                     \
                                        const dim_t* dims,              \
                                        const dim_t* perm,              \
                                        T* b);

  DECLARE_ALL_TYPES(DECLARE_IMPL)

}
