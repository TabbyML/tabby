#pragma once

namespace ctranslate2 {

  enum class Device {
    CPU
  };

#define DEVICE_CASE(DEVICE, STMT)               \
  case DEVICE: {                                \
    const Device D = DEVICE;                    \
    STMT;                                       \
    break;                                      \
  }

#define SINGLE_ARG(...) __VA_ARGS__
#define DEVICE_DISPATCH(DEVICE, STMTS)                  \
  switch (DEVICE) {                                     \
    DEVICE_CASE(Device::CPU, SINGLE_ARG(STMTS))         \
  }

}
