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

        // Backward compatibility with reading the result as a list of log probabilities.
        .def("__len__", [](const ScoringResult& result) {
          return result.tokens_score.size();
        })

        .def("__iter__", [](const ScoringResult& result) {
          return py::make_iterator(result.tokens_score.begin(), result.tokens_score.end());
        })

        .def("__getitem__", [](const ScoringResult& result, size_t i) {
          if (i >= result.tokens_score.size())
            throw py::index_error();
          return result.tokens_score[i];
        })
        ;

      declare_async_wrapper<ScoringResult>(m, "AsyncScoringResult");
    }

  }
}
