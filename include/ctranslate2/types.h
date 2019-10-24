#pragma once

// This file lists supported storage types and statically associates enum entries
// with the true types.
//
// The implemented approach comes from:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.h

#include <string>

namespace ctranslate2 {

  enum class DataType {
    DT_FLOAT,
    DT_INT8,
    DT_INT16,
    DT_INT32
  };

  const std::string& dtype_name(DataType type);

  // Validates type T for whether it is a supported DataType.
  template <class T>
  struct IsValidDataType;

  // DataTypeToEnum<T>::v() and DataTypeToEnum<T>::value are the DataType
  // constants for T, e.g. DataTypeToEnum<float>::v() is DT_FLOAT.
  template <class T>
  struct DataTypeToEnum {
    static_assert(IsValidDataType<T>::value, "Specified Data Type not supported");
  };  // Specializations below

  // EnumToDataType<VALUE>::Type is the type for DataType constant VALUE, e.g.
  // EnumToDataType<DT_FLOAT>::Type is float.
  template <DataType VALUE>
  struct EnumToDataType {}; // Specializations below

#define MATCH_TYPE_AND_ENUM(TYPE, ENUM, NAME)           \
  template<>                                            \
  struct DataTypeToEnum<TYPE> {                         \
    static constexpr DataType value = ENUM;             \
  };                                                    \
  template<>                                            \
  struct IsValidDataType<TYPE> {                        \
    static constexpr bool value = true;                 \
  };                                                    \
  template<>                                            \
  struct EnumToDataType<ENUM> {                         \
    typedef TYPE Type;                                  \
  }

  MATCH_TYPE_AND_ENUM(float, DataType::DT_FLOAT, "float");
  MATCH_TYPE_AND_ENUM(int8_t, DataType::DT_INT8, "int8");
  MATCH_TYPE_AND_ENUM(int16_t, DataType::DT_INT16, "int16");
  MATCH_TYPE_AND_ENUM(int32_t, DataType::DT_INT32, "int32");

  // All types not specialized are marked invalid.
  template <class T>
  struct IsValidDataType {
    static constexpr bool value = false;
  };

#define TYPE_CASE(TYPE, STMTS)                  \
  case DataTypeToEnum<TYPE>::value: {           \
    typedef TYPE T;                             \
    STMTS;                                      \
    break;                                      \
  }

#define SINGLE_ARG(...) __VA_ARGS__
#define TYPE_DISPATCH(TYPE_ENUM, STMTS) \
  switch (TYPE_ENUM) {                  \
    TYPE_CASE(float, SINGLE_ARG(STMTS)) \
    TYPE_CASE(int8_t, SINGLE_ARG(STMTS))            \
    TYPE_CASE(int16_t, SINGLE_ARG(STMTS))           \
    TYPE_CASE(int32_t, SINGLE_ARG(STMTS))           \
  }

#define DECLARE_ALL_TYPES(FUNC)                 \
  FUNC(float)                                   \
  FUNC(int8_t)                                  \
  FUNC(int16_t)                                 \
  FUNC(int32_t)                                 \

}
