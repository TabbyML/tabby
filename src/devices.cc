#include "ctranslate2/devices.h"

#ifdef CT2_WITH_CUDA
#  include "./cuda/utils.h"
#endif

#include "ctranslate2/primitives/primitives.h"
#include "./device_dispatch.h"

namespace ctranslate2 {

  Device str_to_device(const std::string& device) {
#ifdef CT2_WITH_CUDA
    if (device == "cuda" || device == "CUDA")
      return Device::CUDA;
#endif
    if (device == "cpu" || device == "CPU")
      return Device::CPU;
    if (device == "auto" || device == "AUTO")
#ifdef CT2_WITH_CUDA
      return cuda::has_gpu() ? Device::CUDA : Device::CPU;
#else
      return Device::CPU;
#endif
    throw std::invalid_argument("unsupported device " + device);
  }

  std::string device_to_str(Device device) {
    switch (device) {
    case Device::CUDA:
      return "cuda";
    case Device::CPU:
      return "cpu";
    }
    return "";
  }

  template <Device D>
  static void get_and_set_device(const int new_index, int* prev_index) {
    *prev_index = primitives<D>::get_device();
    primitives<D>::set_device(new_index);
  }

  ScopedDeviceSetter::ScopedDeviceSetter(Device device, int index)
    : _device(device) {
    DEVICE_DISPATCH(_device, get_and_set_device<D>(index, &_prev_index));
  }

  ScopedDeviceSetter::~ScopedDeviceSetter() {
    try {
      DEVICE_DISPATCH(_device, primitives<D>::set_device(_prev_index));
    } catch (...) {
    }
  }

}
