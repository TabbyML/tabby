#include <dlfcn.h>
#include <stdexcept>

#include <cublas_v2.h>

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define CUBLAS_SONAME "libcublas.so." STR(CUBLAS_VER_MAJOR)

namespace ctranslate2 {

  static void* get_so_handle() {
    static auto handle = []() {
      void* handle = dlopen(CUBLAS_SONAME, RTLD_LAZY);
      if (!handle)
        throw std::runtime_error(dlerror());
      return handle;
    }();
    return handle;
  }

  template <typename Signature>
  Signature load_symbol(const char* name) {
    void* symbol = dlsym(get_so_handle(), name);
    if (!symbol)
      throw std::runtime_error(dlerror());
    return reinterpret_cast<Signature>(symbol);
  }

}

// TODO: these stub functions could be automatically generated.

extern "C" {

  cublasStatus_t cublasCreate_v2(cublasHandle_t *handle) {
    using Signature = cublasStatus_t(*)(cublasHandle_t*);
    static auto func = ctranslate2::load_symbol<Signature>("cublasCreate_v2");
    return func(handle);
  }

  cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
    using Signature = cublasStatus_t(*)(cublasHandle_t);
    static auto func = ctranslate2::load_symbol<Signature>("cublasDestroy_v2");
    return func(handle);
  }

  cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t stream) {
    using Signature = cublasStatus_t(*)(cublasHandle_t, cudaStream_t);
    static auto func = ctranslate2::load_symbol<Signature>("cublasSetStream_v2");
    return func(handle, stream);
  }

  cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode) {
    using Signature = cublasStatus_t(*)(cublasHandle_t, cublasMath_t*);
    static auto func = ctranslate2::load_symbol<Signature>("cublasGetMathMode");
    return func(handle, mode);
  }

  cublasStatus_t cublasSgemm_v2(cublasHandle_t handle,
                                cublasOperation_t transa,
                                cublasOperation_t transb,
                                int m,
                                int n,
                                int k,
                                const float *alpha,
                                const float *A,
                                int lda,
                                const float *B,
                                int ldb,
                                const float *beta,
                                float *C,
                                int ldc) {
    using Signature = cublasStatus_t(*)(cublasHandle_t,
                                        cublasOperation_t,
                                        cublasOperation_t,
                                        int,
                                        int,
                                        int,
                                        const float*,
                                        const float*,
                                        int,
                                        const float*,
                                        int,
                                        const float*,
                                        float*,
                                        int);
    static auto func = ctranslate2::load_symbol<Signature>("cublasSgemm_v2");
    return func(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle,
                                           cublasOperation_t transa,
                                           cublasOperation_t transb,
                                           int m,
                                           int n,
                                           int k,
                                           const float *alpha,
                                           const float *A,
                                           int lda,
                                           long long int strideA,
                                           const float *B,
                                           int ldb,
                                           long long int strideB,
                                           const float *beta,
                                           float *C,
                                           int ldc,
                                           long long int strideC,
                                           int batchCount) {
    using Signature = cublasStatus_t(*)(cublasHandle_t,
                                        cublasOperation_t,
                                        cublasOperation_t,
                                        int,
                                        int,
                                        int,
                                        const float*,
                                        const float*,
                                        int,
                                        long long int,
                                        const float*,
                                        int,
                                        long long int,
                                        const float*,
                                        float*,
                                        int,
                                        long long int,
                                        int);
    static auto func = ctranslate2::load_symbol<Signature>("cublasSgemmStridedBatched");
    return func(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
  }

  cublasStatus_t cublasGemmEx(cublasHandle_t handle,
                              cublasOperation_t transa,
                              cublasOperation_t transb,
                              int m,
                              int n,
                              int k,
                              const void *alpha,
                              const void *A,
                              cudaDataType Atype,
                              int lda,
                              const void *B,
                              cudaDataType Btype,
                              int ldb,
                              const void *beta,
                              void *C,
                              cudaDataType Ctype,
                              int ldc,
                              cublasComputeType_t computeType,
                              cublasGemmAlgo_t algo) {
    using Signature = cublasStatus_t(*)(cublasHandle_t,
                                        cublasOperation_t,
                                        cublasOperation_t,
                                        int,
                                        int,
                                        int,
                                        const void*,
                                        const void*,
                                        cudaDataType,
                                        int,
                                        const void*,
                                        cudaDataType,
                                        int,
                                        const void*,
                                        void*,
                                        cudaDataType,
                                        int,
                                        cublasComputeType_t,
                                        cublasGemmAlgo_t);
    static auto func = ctranslate2::load_symbol<Signature>("cublasGemmEx");
    return func(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
  }

  cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
                                            cublasOperation_t transa,
                                            cublasOperation_t transb,
                                            int m,
                                            int n,
                                            int k,
                                            const void *alpha,
                                            const void *A,
                                            cudaDataType Atype,
                                            int lda,
                                            long long int strideA,
                                            const void *B,
                                            cudaDataType Btype,
                                            int ldb,
                                            long long int strideB,
                                            const void *beta,
                                            void *C,
                                            cudaDataType Ctype,
                                            int ldc,
                                            long long int strideC,
                                            int batchCount,
                                            cublasComputeType_t computeType,
                                            cublasGemmAlgo_t algo) {
    using Signature = cublasStatus_t(*)(cublasHandle_t,
                                        cublasOperation_t,
                                        cublasOperation_t,
                                        int,
                                        int,
                                        int,
                                        const void*,
                                        const void*,
                                        cudaDataType,
                                        int,
                                        long long int,
                                        const void*,
                                        cudaDataType,
                                        int,
                                        long long int,
                                        const void*,
                                        void*,
                                        cudaDataType,
                                        int,
                                        long long int,
                                        int,
                                        cublasComputeType_t,
                                        cublasGemmAlgo_t);
    static auto func = ctranslate2::load_symbol<Signature>("cublasGemmStridedBatchedEx");
    return func(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
  }

}
