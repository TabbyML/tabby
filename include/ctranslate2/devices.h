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

  int get_device_index(Device device);
  void set_device_index(Device device, int index);

  class ScopedDeviceSetter {
  public:
    ScopedDeviceSetter(Device device, int index)
      : _device(device)
      , _prev_index(get_device_index(device))
    {
      set_device_index(device, index);
    }

    ~ScopedDeviceSetter() {
      // Set previous device index.
      set_device_index(_device, _prev_index);
    }

  private:
    Device _device;
    int _prev_index;
  };

}
