#pragma once

#include <stdexcept>
#include <string>

namespace ctranslate2 {

  enum class Device {
    CPU,
    CUDA
  };

  inline Device str_to_device(const std::string& device) {
#ifdef WITH_CUDA
    if (device == "cuda")
      return Device::CUDA;
#endif
    if (device == "cpu")
      return Device::CPU;
    throw std::invalid_argument("unsupported device " + device);
  }

  inline std::string device_to_str(Device device) {
    switch (device) {
    case Device::CUDA:
      return "CUDA";
    case Device::CPU:
      return "CPU";
    }
    return "";
  }

#define UNSUPPORTED_DEVICE_CASE(DEVICE)                       \
  case DEVICE: {                                              \
    throw std::runtime_error("unsupported device " #DEVICE);  \
    break;                                                    \
  }

#define DEVICE_CASE(DEVICE, STMT)               \
  case DEVICE: {                                \
    const Device D = DEVICE;                    \
    STMT;                                       \
    break;                                      \
  }

#define SINGLE_ARG(...) __VA_ARGS__
#ifndef WITH_CUDA
#  define DEVICE_DISPATCH(DEVICE, STMTS)                \
  switch (DEVICE) {                                     \
    UNSUPPORTED_DEVICE_CASE(Device::CUDA)               \
    DEVICE_CASE(Device::CPU, SINGLE_ARG(STMTS))         \
  }
#else
#  define DEVICE_DISPATCH(DEVICE, STMTS)                \
  switch (DEVICE) {                                     \
    DEVICE_CASE(Device::CUDA, SINGLE_ARG(STMTS))        \
    DEVICE_CASE(Device::CPU, SINGLE_ARG(STMTS))         \
  }
#endif

}
