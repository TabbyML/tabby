#include "module.h"

#include <ctranslate2/translator_pool.h>

namespace ctranslate2 {
  namespace python {

    void register_translation_stats(py::module& m) {
      py::class_<TranslationStats>(m, "TranslationStats",
                                   "A ``namedtuple`` containing some file statistics.")

        .def_readonly("num_tokens", &TranslationStats::num_tokens,
                      "Number of generated tokens.")
        .def_readonly("num_examples", &TranslationStats::num_examples,
                      "Number of processed examples.")
        .def_readonly("total_time_in_ms", &TranslationStats::total_time_in_ms,
                      "Total processing time in milliseconds.")

        .def("__repr__", [](const TranslationStats& stats) {
          return "TranslationStats(num_tokens=" + std::string(py::repr(py::cast(stats.num_tokens)))
            + ", num_examples=" + std::string(py::repr(py::cast(stats.num_examples)))
            + ", total_time_in_ms=" + std::string(py::repr(py::cast(stats.total_time_in_ms)))
            + ")";
        })

        // Backward compatibility with using translate_file output as a tuple.
        .def("__getitem__", [](const TranslationStats& stats, size_t index) {
          auto tuple = py::make_tuple(stats.num_tokens, stats.num_examples, stats.total_time_in_ms);
          return py::object(tuple[index]);
        })
        ;
    }

  }
}
