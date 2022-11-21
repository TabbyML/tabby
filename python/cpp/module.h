#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ctranslate2 {
  namespace python {

    void register_generation_result(py::module& m);
    void register_generator(py::module& m);
    void register_logging(py::module& m);
    void register_scoring_result(py::module& m);
    void register_storage_view(py::module& m);
    void register_translation_result(py::module& m);
    void register_translation_stats(py::module& m);
    void register_translator(py::module& m);
    void register_whisper(py::module& m);

  }
}
