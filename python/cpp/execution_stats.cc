#include "module.h"

#include <ctranslate2/translator.h>

namespace ctranslate2 {
  namespace python {

    void register_translation_stats(py::module& m) {
      py::class_<ExecutionStats>(m, "ExecutionStats",
                                 "A structure containing some execution statistics.")

        .def_readonly("num_tokens", &ExecutionStats::num_tokens,
                      "Number of output tokens.")
        .def_readonly("num_examples", &ExecutionStats::num_examples,
                      "Number of processed examples.")
        .def_readonly("total_time_in_ms", &ExecutionStats::total_time_in_ms,
                      "Total processing time in milliseconds.")

        .def("__repr__", [](const ExecutionStats& stats) {
          return "ExecutionStats(num_tokens=" + std::string(py::repr(py::cast(stats.num_tokens)))
            + ", num_examples=" + std::string(py::repr(py::cast(stats.num_examples)))
            + ", total_time_in_ms=" + std::string(py::repr(py::cast(stats.total_time_in_ms)))
            + ")";
        })
        ;
    }

  }
}
