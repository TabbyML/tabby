#include "module.h"

#include <ctranslate2/scoring.h>

#include "utils.h"

namespace ctranslate2 {
  namespace python {

    void register_scoring_result(py::module& m) {
      py::class_<ScoringResult>(m, "ScoringResult", "A scoring result.")

        .def_readonly("tokens", &ScoringResult::tokens,
                      "The scored tokens.")
        .def_readonly("log_probs", &ScoringResult::tokens_score,
                      "Log probability of each token")

        .def("__repr__", [](const ScoringResult& result) {
          return "ScoringResult(tokens=" + std::string(py::repr(py::cast(result.tokens)))
            + ", log_probs=" + std::string(py::repr(py::cast(result.tokens_score)))
            + ")";
        })
        ;

      declare_async_wrapper<ScoringResult>(m, "AsyncScoringResult");
    }

  }
}
