#include "module.h"

#include <ctranslate2/logging.h>

namespace ctranslate2 {
  namespace python {

    void register_logging(py::module& m) {
      py::enum_<ctranslate2::LogLevel>(m, "LogLevel")
        .value("Off", ctranslate2::LogLevel::Off)
        .value("Critical", ctranslate2::LogLevel::Critical)
        .value("Error", ctranslate2::LogLevel::Error)
        .value("Warning", ctranslate2::LogLevel::Warning)
        .value("Info", ctranslate2::LogLevel::Info)
        .value("Debug", ctranslate2::LogLevel::Debug)
        .value("Trace", ctranslate2::LogLevel::Trace)
        .export_values();

      m.def("set_log_level", &ctranslate2::set_log_level);
      m.def("get_log_level", &ctranslate2::get_log_level);
    }

  }
}
