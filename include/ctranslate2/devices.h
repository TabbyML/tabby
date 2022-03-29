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

  int get_gpu_count();

  void synchronize_device(Device device, int index);

  class ScopedDeviceSetter {
  public:
    ScopedDeviceSetter(Device device, int index)
      : _device(device)
      , _prev_index(get_device_index(device))
      , _new_index(index)
    {
      if (_prev_index != _new_index)
        set_device_index(device, _new_index);
    }

    ~ScopedDeviceSetter() {
      // Set previous device index.
      if (_prev_index != _new_index)
        set_device_index(_device, _prev_index);
    }

  private:
    Device _device;
    int _prev_index;
    int _new_index;
  };

}
