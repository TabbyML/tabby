#pragma once

#include "device_dispatch.h"
#include "type_dispatch.h"

#define DEVICE_AND_TYPE_DISPATCH(DEVICE, TYPE, STMTS)   \
  DEVICE_DISPATCH(DEVICE, TYPE_DISPATCH(TYPE, (STMTS)))


#define NON_FLOAT_CASE(NAME)                                            \
  default:                                                              \
    throw std::invalid_argument(NAME " only supports float types");     \


#ifndef CT2_WITH_CUDA

#  define DEVICE_AND_FLOAT_DISPATCH(NAME, DEVICE, TYPE, STMTS)          \
  switch (TYPE) {                                                       \
    TYPE_CASE(float, DEVICE_DISPATCH(DEVICE, (STMTS)))                  \
    NON_FLOAT_CASE(NAME)                                                \
  }

#else

#  define DEVICE_AND_FLOAT_DISPATCH(NAME, DEVICE, TYPE, STMTS)          \
  switch (TYPE) {                                                       \
    TYPE_CASE(float, DEVICE_DISPATCH(DEVICE, (STMTS)))                  \
    TYPE_CASE(float16_t, {                                              \
      if (DEVICE != Device::CUDA)                                       \
        throw std::invalid_argument("FP16 " NAME " is only supported on GPU"); \
      constexpr Device D = Device::CUDA;                                \
      (STMTS);                                                          \
    })                                                                  \
    TYPE_CASE(bfloat16_t, {                                             \
      if (DEVICE != Device::CUDA)                                       \
        throw std::invalid_argument("BF16 " NAME " is only supported on GPU"); \
      constexpr Device D = Device::CUDA;                                \
      (STMTS);                                                          \
    })                                                                  \
    NON_FLOAT_CASE(NAME)                                                \
  }

#endif
