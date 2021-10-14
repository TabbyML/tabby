#include "ctranslate2/ops/gemm.h"

#include "ctranslate2/ops/bias_add.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void apply_bias_and_activation(StorageView& x,
                                   const StorageView* bias,
                                   const ActivationType* activation_type) {
      if (bias) {
        const ops::BiasAdd bias_add_op(activation_type);
        bias_add_op(x, *bias, x);
      } else if (activation_type) {
        get_activation_op(*activation_type)(x, x);
      }
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
                          StorageView& c,
                          const StorageView* a_shift_compensation,
                          const StorageView* bias) const {
      PROFILE("Gemm");

      switch (a.dtype()) {
      case DataType::INT8:
        DEVICE_DISPATCH(a.device(), (compute<D, int8_t, int32_t>(a, b, c, a_shift_compensation)));
        break;

      case DataType::INT16:
        if (a.device() != Device::CPU)
          throw std::invalid_argument("INT16 GEMM is only supported on CPU");
        compute<Device::CPU, int16_t, int32_t>(a, b, c, a_shift_compensation);
        break;

      case DataType::FLOAT:
        DEVICE_DISPATCH(a.device(), (compute<D, float, float>(a, b, c, a_shift_compensation)));
        break;

#ifdef CT2_WITH_CUDA
      case DataType::FLOAT16:
        if (a.device() != Device::CUDA)
          throw std::invalid_argument("FP16 GEMM is only supported on GPU");
        compute<Device::CUDA, float16_t, float16_t>(a, b, c, a_shift_compensation);
        break;
#endif

      default:
        throw std::invalid_argument("unsupported compute type " + dtype_name(a.dtype()));
      }

      apply_bias_and_activation(c, bias, _activation_type);
    }

    template <Device D, typename In, typename Out>
    void Gemm::compute(const StorageView& a,
                       const StorageView& b,
                       StorageView& c,
                       const StorageView* a_shift_compensation) const {
      const dim_t k = a.dim(_trans_a ? -2 : -1);
      const dim_t n = b.dim(_trans_b ? -2 : -1);
      const dim_t m = a.size() / k;  // Collapse leading dimensions.
      const dim_t lda = _trans_a ? m : k;
      const dim_t ldb = _trans_b ? k : n;
      const dim_t ldc = n;

      {
        Shape output_shape(a.shape());
        output_shape[output_shape.size() - 1] = n;
        c.resize(std::move(output_shape));
      }

      primitives<D>::gemm(_a_is_packed, _b_is_packed,
                          _trans_a, _trans_b,
                          m, n, k,
                          _alpha,
                          a.data<In>(), lda,
                          b.data<In>(), ldb,
                          _beta,
                          c.data<Out>(), ldc,
                          a_shift_compensation ? a_shift_compensation->data<Out>() : nullptr);
    }

  }
}
