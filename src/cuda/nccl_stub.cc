#include <stdexcept>

#include <nccl.h>

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#include <dlfcn.h>
#define NCCL_LIBNAME "libnccl.so." STR(NCCL_MAJOR)

#include <spdlog/spdlog.h>

namespace ctranslate2 {

  template <typename Signature>
  static Signature load_symbol(void* handle, const char* name, const char* library_name) {
    void* symbol = dlsym(handle, name);
    if (!symbol)
      throw std::runtime_error("Cannot load symbol " + std::string(name)
                               + " from library " + std::string(library_name));
    return reinterpret_cast<Signature>(symbol);
  }
  static inline void log_nccl_version(void* handle) {
    using Signature = ncclResult_t(*)(int*);
    const auto nccl_get_version = load_symbol<Signature>(handle,
                                                            "ncclGetVersion",
                                                            NCCL_LIBNAME);
    int version = 0;
    nccl_get_version(&version);
    spdlog::info("Loaded nccl library version {}", version);
  }

  static void* get_so_handle() {
    static auto so_handle = []() {
      void* handle = dlopen(NCCL_LIBNAME, RTLD_LAZY);
      if (!handle)
        throw std::runtime_error("Library " + std::string(NCCL_LIBNAME)
                                 + " is not found or cannot be loaded");
      log_nccl_version(handle);
      return handle;
    }();
    return so_handle;
  }

  template <typename Signature>
  static Signature load_symbol(const char* name) {
    void* handle = get_so_handle();
    return load_symbol<Signature>(handle, name, NCCL_LIBNAME);
  }

}

extern "C" {
  ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
    using Signature = ncclResult_t(*)(ncclUniqueId* uniqueId);
    static auto func = ctranslate2::load_symbol<Signature>("ncclGetUniqueId");
    return func(uniqueId);
  }

  ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
    using Signature = ncclResult_t(*)(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
    static auto func = ctranslate2::load_symbol<Signature>("ncclCommInitRank");
    return func(comm, nranks, commId, rank);
  }

  ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  using Signature = ncclResult_t(*)(ncclComm_t comm);
  static auto func = ctranslate2::load_symbol<Signature>("ncclCommDestroy");
  return func(comm);
  }

  ncclResult_t ncclCommFinalize(ncclComm_t comm) {
    using Signature = ncclResult_t(*)(ncclComm_t comm);
    static auto func = ctranslate2::load_symbol<Signature>("ncclCommFinalize");
    return func(comm);
  }

  ncclResult_t  ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                              ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
    using Signature = ncclResult_t(*)(const void* sendbuff, void* recvbuff, size_t count,
                                      ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
    static auto func = ctranslate2::load_symbol<Signature>("ncclAllReduce");
    return func(sendbuff, recvbuff, count, datatype, op, comm, stream);
  }

  ncclResult_t  ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                              ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
    using Signature = ncclResult_t(*)(const void* sendbuff, void* recvbuff, size_t sendcount,
                                      ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
    static auto func = ctranslate2::load_symbol<Signature>("ncclAllGather");
    return func(sendbuff, recvbuff, sendcount, datatype, comm, stream);
  }
}
