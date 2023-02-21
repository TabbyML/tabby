#include <pybind11/pybind11.h>

#include <ctranslate2/devices.h>
#include <ctranslate2/models/model.h>
#include <ctranslate2/random.h>
#include <ctranslate2/types.h>
#include <ctranslate2/utils.h>

#include "module.h"
#include "utils.h"


static std::unordered_set<std::string>
get_supported_compute_types(const std::string& device_str, const int device_index) {
  const auto device = ctranslate2::str_to_device(device_str);

  const bool support_float16 = ctranslate2::mayiuse_float16(device, device_index);
  const bool support_int16 = ctranslate2::mayiuse_int16(device, device_index);
  const bool support_int8 = ctranslate2::mayiuse_int8(device, device_index);

  std::unordered_set<std::string> compute_types;
  compute_types.emplace("float32");
  if (support_float16)
    compute_types.emplace("float16");
  if (support_int16)
    compute_types.emplace("int16");
  if (support_int8)
    compute_types.emplace("int8");
  if (support_int8 && support_float16)
    compute_types.emplace("int8_float16");
  return compute_types;
}


PYBIND11_MODULE(_ext, m)
{
  m.def("contains_model", &ctranslate2::models::contains_model, py::arg("path"),
        "Helper function to check if a directory seems to contain a CTranslate2 model.");

  m.def("get_cuda_device_count", &ctranslate2::get_gpu_count,
        "Returns the number of visible GPU devices.");

  m.def("get_supported_compute_types", &get_supported_compute_types,
        py::arg("device"),
        py::arg("device_index")=0,
         R"pbdoc(
             Returns the set of supported compute types on a device.

             Arguments:
               device: Device name (cpu or cuda).
               device_index: Device index.

             Example:
                 >>> ctranslate2.get_supported_compute_types("cpu")
                 {'int16', 'float32', 'int8'}
                 >>> ctranslate2.get_supported_compute_types("cuda")
                 {'float32', 'int8_float16', 'float16', 'int8'}
         )pbdoc");

  m.def("set_random_seed", &ctranslate2::set_random_seed, py::arg("seed"),
        "Sets the seed of random generators.");

  ctranslate2::python::register_logging(m);
  ctranslate2::python::register_storage_view(m);
  ctranslate2::python::register_translation_stats(m);
  ctranslate2::python::register_translation_result(m);
  ctranslate2::python::register_scoring_result(m);
  ctranslate2::python::register_generation_result(m);
  ctranslate2::python::register_translator(m);
  ctranslate2::python::register_generator(m);
  ctranslate2::python::register_whisper(m);
}
