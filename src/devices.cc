#include "ctranslate2/devices.h"

namespace ctranslate2 {

  Device str_to_device(const std::string& device) {
#ifdef WITH_CUDA
    if (device == "cuda" || device == "CUDA")
      return Device::CUDA;
#endif
    if (device == "cpu" || device == "CPU")
      return Device::CPU;
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
