#pragma once

#include <stdexcept>
#include <string>

namespace ctranslate2 {

  enum class ComputeType {
    DEFAULT,
    FLOAT,
    INT8,
    INT16
  };

  ComputeType str_to_compute_type(const std::string& compute_type);

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
