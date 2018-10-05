#include "ctranslate2/devices.h"

#ifdef WITH_CUDA
#  include "ctranslate2/cuda/utils.h"
#endif

namespace ctranslate2 {

  Device str_to_device(const std::string& device) {
#ifdef WITH_CUDA
    if (device == "cuda" || device == "CUDA")
      return Device::CUDA;
#endif
    if (device == "cpu" || device == "CPU")
      return Device::CPU;
    if (device == "auto" || device == "AUTO")
#ifdef WITH_CUDA
      return cuda::get_gpu_count() > 0 ? Device::CUDA : Device::CPU;
#else
      return Device::CPU;
#endif
    throw std::invalid_argument("unsupported device " + device);
  }

  std::string device_to_str(Device device) {
    switch (device) {
    case Device::CUDA:
      return "CUDA";
    case Device::CPU:
      return "CPU";
    }
    return "";
  }

}
