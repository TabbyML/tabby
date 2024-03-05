#include <stdexcept>
#include <cuda/mpi_stub.h>

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#include <dlfcn.h>

#define OPENMPI_LIBNAME "libmpi.so." STR(OMPI_MAJOR_VERSION) STR(0)

namespace ctranslate2 {

  template <typename Signature>
  static Signature load_symbol(void* handle, const char* name, const char* library_name) {
    void* symbol = dlsym(handle, name);
    if (!symbol)
      throw std::runtime_error("Cannot load symbol " + std::string(name)
                               + " from library " + std::string(library_name));
    return reinterpret_cast<Signature>(symbol);
  }

  static void* get_so_handle() {
    static auto so_handle = []() {
      void* handle = dlopen(OPENMPI_LIBNAME, RTLD_LAZY);
      return handle;
    }();
    return so_handle;
  }

  template <typename Signature>
  static Signature load_symbol(const char* name) {
    void* handle = get_so_handle();
    if (!handle)
      throw std::runtime_error("Library " + std::string(OPENMPI_LIBNAME)
                               + " is not found or cannot be loaded");
    return load_symbol<Signature>(handle, name, OPENMPI_LIBNAME);
  }

  template <typename Signature>
  static Signature load_symbol_global(const char* name) {
    void* handle = get_so_handle();
    if (!handle)
      return nullptr;
    return load_symbol<Signature>(handle, name, OPENMPI_LIBNAME);
  }
}

extern "C" {

  int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, int recvcount,
                    MPI_Datatype recvtype, MPI_Comm comm) {
    using Signature = int(*)(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount,
                             MPI_Datatype recvtype, MPI_Comm comm);
    static auto func = ctranslate2::load_symbol<Signature>("MPI_Allgather");
    return func(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
  }

  int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
              int root, MPI_Comm comm) {
    using Signature = int(*)(void *buffer, int count, MPI_Datatype datatype,
                             int root, MPI_Comm comm);
    static auto func = ctranslate2::load_symbol<Signature>("MPI_Bcast");
    return func(buffer, count, datatype, root, comm);
  }

  int MPI_Init(int *argc, char ***argv) {
    using Signature = int(*)(int *argc, char ***argv);
    static auto func = ctranslate2::load_symbol<Signature>("MPI_Init");
    return func(argc, argv);
  }

  int MPI_Finalize(void) {
    using Signature = int(*)(void);
    static auto func = ctranslate2::load_symbol<Signature>("MPI_Finalize");
    return func();
  }

  int MPI_Comm_rank(MPI_Comm comm, int *rank) {
    using Signature = int(*)(MPI_Comm comm, int *size);
    static auto func = ctranslate2::load_symbol<Signature>("MPI_Comm_rank");
    return func(comm, rank);
  }

  int MPI_Comm_size(MPI_Comm comm, int *size) {
    using Signature = int(*)(MPI_Comm comm, int *size);
    static auto func = ctranslate2::load_symbol<Signature>("MPI_Comm_size");
    return func(comm, size);
  }
}
struct ompi_predefined_datatype_t* stub_mpi_datatype_null = ctranslate2::load_symbol_global<struct ompi_predefined_datatype_t*>("ompi_mpi_datatype_null");
struct ompi_predefined_datatype_t* stub_ompi_mpi_byte = ctranslate2::load_symbol_global<struct ompi_predefined_datatype_t*>("ompi_mpi_byte");
struct ompi_predefined_communicator_t* stub_ompi_mpi_comm_world = ctranslate2::load_symbol_global<struct ompi_predefined_communicator_t*>("ompi_mpi_comm_world");