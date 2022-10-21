#include "module.h"

#include <ctranslate2/translation.h>

#include "utils.h"

namespace ctranslate2 {
  namespace python {

    void register_translation_result(py::module& m) {
      py::class_<TranslationResult>(m, "TranslationResult", "A translation result.")

        .def_readonly("hypotheses", &TranslationResult::hypotheses,
                      "Translation hypotheses.")
        .def_readonly("scores", &TranslationResult::scores,
                      "Score of each translation hypothesis (empty if :obj:`return_scores` was disabled).")
        .def_readonly("attention", &TranslationResult::attention,
                      "Attention matrix of each translation hypothesis (empty if :obj:`return_attention` was disabled).")

        .def("__repr__", [](const TranslationResult& result) {
          return "TranslationResult(hypotheses=" + std::string(py::repr(py::cast(result.hypotheses)))
            + ", scores=" + std::string(py::repr(py::cast(result.scores)))
            + ", attention=" + std::string(py::repr(py::cast(result.attention)))
            + ")";
        })

        // Backward compatibility with using translate_batch output as a list of dicts.
        .def("__len__", &TranslationResult::num_hypotheses)

        .def("__getitem__", [](const TranslationResult& result, size_t i) {
          if (i >= result.num_hypotheses())
            throw py::index_error();
          py::dict hypothesis;
          hypothesis["tokens"] = result.hypotheses[i];
          if (result.has_scores())
            hypothesis["score"] = result.scores[i];
          if (result.has_attention())
            hypothesis["attention"] = result.attention[i];
          return hypothesis;
        })
        ;

      declare_async_wrapper<TranslationResult>(m, "AsyncTranslationResult");
    }

  }
}
