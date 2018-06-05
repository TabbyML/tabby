#pragma once

#include <gtest/gtest.h>

#include "opennmt/storage_view.h"

using namespace opennmt;

template <typename T>
void expect_array_eq(const T* x, const T* y, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    EXPECT_EQ(x[i], y[i]) << "Value mismatch at index " << i;
  }
}

template <typename T>
void assert_vector_eq(const std::vector<T>& got, const std::vector<T>& expected) {
  ASSERT_EQ(got.size(), expected.size());
  for (size_t i = 0; i < got.size(); ++i) {
    ASSERT_EQ(got[i], expected[i]) << "Value mismatch for dimension " << i;
  }
}

void expect_storage_eq(const StorageView& got, const StorageView& expected) {
  ASSERT_EQ(got.dtype(), expected.dtype());
  assert_vector_eq(got.shape(), expected.shape());
  TYPE_DISPATCH(got.dtype(), expect_array_eq(got.data<T>(), expected.data<T>(), got.size()));
}
