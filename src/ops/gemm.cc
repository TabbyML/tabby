#include "ctranslate2/ops/gemm.h"

#include "device_dispatch.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void add_bias(StorageView& x, const StorageView& bias) {
      DEVICE_DISPATCH(x.device(),
                      TYPE_DISPATCH(bias.dtype(),
                                    primitives<D>::add_batch_broadcast(bias.data<T>(),
                                                                       x.data<T>(),
                                                                       bias.size(),
                                                                       x.size())));
    }

    template <Device D, typename In, typename Out>
    static void run_gemm(const StorageView& a, const StorageView& b,
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
               bool b_is_packed,
               const ActivationType* activation_type)
      : _alpha(alpha)
      , _beta(beta)
      , _trans_a(trans_a)
      , _trans_b(trans_b)
      , _a_is_packed(a_is_packed)
      , _b_is_packed(b_is_packed)
      , _activation_type(activation_type)
    {
    }

    void Gemm::operator()(const StorageView& a,
                          const StorageView& b,
                          StorageView& y,
                          const StorageView* a_shift_compensation,
                          const StorageView* bias) const {
      PROFILE("Gemm");

      const dim_t k = a.dim(_trans_a ? -2 : -1);
      const dim_t n = b.dim(_trans_b ? -2 : -1);
      const dim_t m = a.size() / k;  // Collapse leading dimensions.

      Shape output_shape(a.shape());
      output_shape[output_shape.size() - 1] = n;
      y.resize(std::move(output_shape));

      switch (a.dtype()) {
      case DataType::INT8:
        DEVICE_DISPATCH(a.device(),
                        (run_gemm<D, int8_t, int32_t>(a, b,
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
        run_gemm<Device::CPU, int16_t, int32_t>(a, b,
                                                _a_is_packed, _b_is_packed,
                                                _trans_a, _trans_b,
                                                m, n, k,
                                                _alpha, _beta,
                                                y,
                                                a_shift_compensation);
        break;

      case DataType::FLOAT:
        DEVICE_DISPATCH(a.device(),
                        (run_gemm<D, float, float>(a, b,
                                                   _a_is_packed, _b_is_packed,
                                                   _trans_a, _trans_b,
                                                   m, n, k,
                                                   _alpha, _beta,
                                                   y,
                                                   a_shift_compensation)));
        break;

#ifdef CT2_WITH_CUDA
      case DataType::FLOAT16:
        if (a.device() != Device::CUDA)
          throw std::invalid_argument("FP16 GEMM is only supported on GPU");
        run_gemm<Device::CUDA, float16_t, float16_t>(a, b,
                                                     _a_is_packed, _b_is_packed,
                                                     _trans_a, _trans_b,
                                                     m, n, k,
                                                     _alpha, _beta,
                                                     y,
                                                     a_shift_compensation);
        break;
#endif

      default:
        throw std::invalid_argument("unsupported compute type " + dtype_name(a.dtype()));
      }

      if (bias)
        add_bias(y, *bias);
      if (_activation_type)
        get_activation_op(*_activation_type)(y, y);
    }

  }
}
