#include "module.h"

#include <ctranslate2/devices.h>

#include "utils.h"

namespace ctranslate2 {
  namespace python {

    void register_mpi(py::module& m) {
      py::class_<ScopedMPISetter>(
        m, "MpiInfo",
        R"pbdoc(
            An object to manage the MPI communication between processes.
            It provides information about MPI connexion.
        )pbdoc")

        .def_static("getNRanks", &ScopedMPISetter::getNRanks,
                             "Get the number of gpus running for the current model.")

        .def_static("getCurRank", &ScopedMPISetter::getCurRank,
                             "Get the current rank of process.")

        .def_static("getLocalRank", &ScopedMPISetter::getLocalRank,
                             "Get the current GPU id used by process.")
        ;
    }

  }
}
