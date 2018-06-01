#pragma once

#include "compute.h"
#include "storage_view.h"

static void swap_middle_dims(const onmt::StorageView& x, onmt::StorageView& y) {
  assert(x.rank() == 4);
  size_t d0 = x.dim(0);
  size_t d1 = x.dim(1);
  size_t d2 = x.dim(2);
  size_t d3 = x.dim(3);
  y.resize({d0, d2, d1, d3});
  for (size_t i0 = 0; i0 < d0; ++i0) {
    for (size_t i1 = 0; i1 < d1; ++i1) {
      for (size_t i2 = 0; i2 < d2; ++i2) {
        for (size_t i3 = 0; i3 < d3; ++i3) {
          y.data<float>()[i3 + (i1 * d3) + (i2 * d3 * d1) + (i0 * d3 * d1 * d2)] =
            x.data<float>()[i3 + (i2 * d3) + (i1 * d3 * d2) + (i0 * d3 * d2 * d1)];
        }
      }
    }
  }
}
