#include "module.h"

#include <ctranslate2/generation.h>

#include "utils.h"

namespace ctranslate2 {
  namespace python {

    void register_generation_result(py::module& m) {
      py::class_<GenerationResult>(m, "GenerationResult", "A generation result.")

        .def_readonly("sequences", &GenerationResult::sequences,
                      "Generated sequences of tokens.")
        .def_readonly("sequences_ids", &GenerationResult::sequences_ids,
                      "Generated sequences of token IDs.")
        .def_readonly("scores", &GenerationResult::scores,
                      "Score of each sequence (empty if :obj:`return_scores` was disabled).")

        .def("__repr__", [](const GenerationResult& result) {
          return "GenerationResult(sequences=" + std::string(py::repr(py::cast(result.sequences)))
            + ", sequences_ids=" + std::string(py::repr(py::cast(result.sequences_ids)))
            + ", scores=" + std::string(py::repr(py::cast(result.scores)))
            + ")";
        })
        ;

      declare_async_wrapper<GenerationResult>(m, "AsyncGenerationResult");
    }

  }
}
