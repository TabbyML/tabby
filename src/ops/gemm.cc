#include "ctranslate2/ops/gemm.h"

#include "../device_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename In, typename Out>
    static void run_gemm(const StorageView& a, const StorageView& b, const StorageView* c,
                         bool a_is_packed, bool b_is_packed,
                         bool transpose_a, bool transpose_b,
                         dim_t m, dim_t n, dim_t k,
                         float alpha, float beta,
                         StorageView& y,
                         const StorageView* a_shift_compensation = nullptr) {
      const In* a_data = a.data<In>();
      const In* b_data = b.data<In>();
      Out* y_data = y.data<Out>();
      const Out* a_shift_compensation_data = (a_shift_compensation
                                              ? a_shift_compensation->data<Out>()
                                              : nullptr);

      if (beta != 0 && c) {
        const Out* c_data = c->data<Out>();
        if (c->size() == y.size()) {
          primitives<D>::copy(c_data, y_data, y.size());
        } else if (c->size() == n) {
          for (dim_t i = 0; i < m; ++i)
            primitives<D>::copy(c_data, y_data + i * n, n);
        } else {
          throw std::invalid_argument("c has invalid size");
        }
      }

      primitives<D>::gemm(a_data, b_data,
                          a_is_packed, b_is_packed,
                          transpose_a, transpose_b,
                          m, n, k,
                          alpha, beta,
                          y_data,
                          a_shift_compensation_data);
    }


    Gemm::Gemm(float alpha,
               float beta,
               bool trans_a,
               bool trans_b,
               bool a_is_packed,
               bool b_is_packed)
      : _alpha(alpha)
      , _beta(beta)
      , _trans_a(trans_a)
      , _trans_b(trans_b)
      , _a_is_packed(a_is_packed)
      , _b_is_packed(b_is_packed) {
    }

    void Gemm::operator()(const StorageView& a,
                          const StorageView& b,
                          const StorageView& c,
                          StorageView& y) const {
      compute(a, b, &c, y, nullptr);
    }

    void Gemm::operator()(const StorageView& a,
                          const StorageView& b,
                          StorageView& c,
                          const StorageView* a_shift_compensation) const {
      compute(a, b, nullptr, c, a_shift_compensation);
    }

    void Gemm::compute(const StorageView& a,
                       const StorageView& b,
                       const StorageView* c,
                       StorageView& y,
                       const StorageView* a_shift_compensation) const {
      PROFILE("Gemm");

      const dim_t k = a.dim(_trans_a ? -2 : -1);
      const dim_t n = b.dim(_trans_b ? -2 : -1);
      const dim_t m = a.size() / k;  // Collapse leading dimensions.

      Shape output_shape(a.shape());
      output_shape[output_shape.size() - 1] = n;
      y.resize(output_shape);

      switch (a.dtype()) {
      case DataType::INT8:
        DEVICE_DISPATCH(a.device(),
                        (run_gemm<D, int8_t, int32_t>(a, b, c,
                                                      _a_is_packed, _b_is_packed,
                                                      _trans_a, _trans_b,
                                                      m, n, k,
                                                      _alpha, _beta,
                                                      y,
                                                      a_shift_compensation)));
        break;

      case DataType::INT16:
        if (a.device() != Device::CPU)
          throw std::invalid_argument("INT16 GEMM is only supported on CPU");
        run_gemm<Device::CPU, int16_t, int32_t>(a, b, c,
                                                _a_is_packed, _b_is_packed,
                                                _trans_a, _trans_b,
                                                m, n, k,
                                                _alpha, _beta,
                                                y,
                                                a_shift_compensation);
        break;

      case DataType::FLOAT:
        DEVICE_DISPATCH(a.device(),
                        (run_gemm<D, float, float>(a, b, c,
                                                   _a_is_packed, _b_is_packed,
                                                   _trans_a, _trans_b,
                                                   m, n, k,
                                                   _alpha, _beta,
                                                   y,
                                                   a_shift_compensation)));
        break;

      default:
        throw std::invalid_argument("unsupported compute type " + dtype_name(a.dtype()));
      }
    }

  }
}
