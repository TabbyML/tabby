#pragma once

#include <mpi.h>

#ifdef CT2_WITH_CUDA_DYNAMIC_LOADING
extern struct ompi_predefined_datatype_t* stub_mpi_datatype_null;
#define STUB_MPI_DATATYPE_NULL OMPI_PREDEFINED_GLOBAL(MPI_Datatype, *stub_mpi_datatype_null)

extern struct ompi_predefined_datatype_t* stub_ompi_mpi_byte;
#define STUB_MPI_BYTE OMPI_PREDEFINED_GLOBAL(MPI_Datatype, *stub_ompi_mpi_byte)

extern struct ompi_predefined_communicator_t* stub_ompi_mpi_comm_world;
#define STUB_MPI_COMM_WORLD OMPI_PREDEFINED_GLOBAL(MPI_Comm, *stub_ompi_mpi_comm_world)
#else
#define STUB_MPI_DATATYPE_NULL MPI_DATATYPE_NULL
#define STUB_MPI_BYTE MPI_BYTE
#define STUB_MPI_COMM_WORLD MPI_COMM_WORLD
#endif