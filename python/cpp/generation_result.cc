#include "module.h"

#include <ctranslate2/generation.h>

#include "utils.h"

namespace ctranslate2 {
  namespace python {

    void register_generation_result(py::module& m) {
      py::class_<GenerationStepResult>(m, "GenerationStepResult",
                                       "The result for a single generation step.")

        .def_readonly("step", &GenerationStepResult::step,
                      "The decoding step.")
        .def_readonly("batch_id", &GenerationStepResult::batch_id,
                      "The batch index.")
        .def_readonly("token_id", &GenerationStepResult::token_id,
                      "ID of the generated token.")
        .def_readonly("hypothesis_id", &GenerationStepResult::hypothesis_id,
                      "Index of the hypothesis in the batch.")
        .def_readonly("token", &GenerationStepResult::token,
                      "String value of the generated token.")
        .def_readonly("log_prob", &GenerationStepResult::log_prob,
                      "Log probability of the token (``None`` if :obj:`return_log_prob` was disabled).")
        .def_readonly("is_last", &GenerationStepResult::is_last,
                      "Whether this step is the last decoding step for this batch.")

        .def("__repr__", [](const GenerationStepResult& result) {
          return "GenerationStepResult(step=" + std::string(py::repr(py::cast(result.step)))
            + ", batch_id=" + std::string(py::repr(py::cast(result.batch_id)))
            + ", token_id=" + std::string(py::repr(py::cast(result.token_id)))
            + ", hypothesis_id=" + std::string(py::repr(py::cast(result.hypothesis_id)))
            + ", token=" + std::string(py::repr(py::cast(result.token)))
            + ", log_prob=" + std::string(py::repr(py::cast(result.log_prob)))
            + ", is_last=" + std::string(py::repr(py::cast(result.is_last)))
            + ")";
        })
        ;

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
