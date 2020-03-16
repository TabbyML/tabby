#pragma once

#include <stdexcept>
#include <string>

namespace ctranslate2 {

  enum class Device {
    CPU,
    CUDA
  };

  Device str_to_device(const std::string& device);
  std::string device_to_str(Device device);

  class ScopedDeviceSetter {
  public:
    ScopedDeviceSetter(Device device, int index);
    ~ScopedDeviceSetter();  // Set previous device index.

  private:
    Device _device;
    int _prev_index;
  };

}
