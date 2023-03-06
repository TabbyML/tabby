#include "cpu/kernels.h"

#include <limits>

#if defined(__AVX512F__)
#  define TARGET_ISA CpuIsa::AVX512
#  include "cpu/vec_avx512.h"
#elif defined(__AVX2__)
#  define TARGET_ISA CpuIsa::AVX2
#  include "cpu/vec_avx.h"
#elif defined(__AVX__)
#  define TARGET_ISA CpuIsa::AVX
#  include "cpu/vec_avx.h"
#elif (defined(__ARM_NEON) && !defined(CT2_WITH_CPU_DISPATCH)) || defined(USE_NEON)
#  define TARGET_ISA CpuIsa::NEON
#  include "cpu/vec_neon.h"
#else
#  define TARGET_ISA CpuIsa::GENERIC
#  include "cpu/vec.h"
#endif

// Associative arithmetic is required for the compiler to automatically vectorize loops.
// With GCC this can be minimally enabled with -fassociative-math but this flag is not
// available in other compiler.
//
// The macros below can be used to enabled fast-math locally.
//
// See https://github.com/marian-nmt/marian-dev/blob/1.11.0/src/common/definitions.h#L25-L78
// for more details.
#if defined(_MSC_VER)
#  define CT2_FFAST_MATH_BEGIN __pragma(float_control(precise, off, push))
#  define CT2_FFAST_MATH_END __pragma(float_control(pop))
#elif defined(__clang__)
#  define CT2_FFAST_MATH_BEGIN _Pragma("float_control(precise, off, push)")
#  define CT2_FFAST_MATH_END _Pragma("float_control(pop)")
#elif defined(__GNUC__)
#  define CT2_FFAST_MATH_BEGIN _Pragma("GCC push_options") _Pragma("GCC optimize(\"-ffast-math\")")
#  define CT2_FFAST_MATH_END _Pragma("GCC pop_options")
#endif

#include "cpu/parallel.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace cpu {

    template <CpuIsa ISA, typename T, typename Function>
    static void vectorized_unary_transform(const T* x, T* y, dim_t size, const Function& func) {
      const dim_t remaining = size % Vec<T, ISA>::width;
      size -= remaining;

      for (dim_t i = 0; i < size; i += Vec<T, ISA>::width) {
        auto v = Vec<T, ISA>::load(x + i);
        Vec<T, ISA>::store(func(v), y + i);
      }

      if (remaining != 0) {
        auto v = Vec<T, ISA>::load(x + size, remaining);
        Vec<T, ISA>::store(func(v), y + size, remaining);
      }
    }

    template <CpuIsa ISA, typename T, typename Function>
    static void vectorized_binary_transform(const T* a,
                                            const T* b,
                                            T* c,
                                            dim_t size,
                                            const Function& func) {
      const dim_t remaining = size % Vec<T, ISA>::width;
      size -= remaining;

      for (dim_t i = 0; i < size; i += Vec<T, ISA>::width) {
        auto v1 = Vec<T, ISA>::load(a + i);
        auto v2 = Vec<T, ISA>::load(b + i);
        Vec<T, ISA>::store(func(v1, v2), c + i);
      }

      if (remaining != 0) {
        auto v1 = Vec<T, ISA>::load(a + size, remaining);
        auto v2 = Vec<T, ISA>::load(b + size, remaining);
        Vec<T, ISA>::store(func(v1, v2), c + size, remaining);
      }
    }

    template <CpuIsa ISA,
              typename T,
              typename VecMapFunc,
              typename VecReduceFunc,
              typename VecHorizontalReduceFunc,
              typename ScalarMapFunc,
              typename ScalarReduceFunc>
    static T vectorized_map_reduce_all(const T* x,
                                       dim_t size,
                                       T init,
                                       const VecMapFunc& vec_map_func,
                                       const VecReduceFunc& vec_reduce_func,
                                       const VecHorizontalReduceFunc& vec_horizontal_reduce_func,
                                       const ScalarMapFunc& scalar_map_func,
                                       const ScalarReduceFunc& scalar_reduce_func) {
      if (Vec<T, ISA>::width == 1 || size <= Vec<T, ISA>::width) {
        T accu = init;
        for (dim_t i = 0; i < size; ++i) {
          accu = scalar_reduce_func(accu, scalar_map_func(x[i]));
        }
        return accu;
      }

      const dim_t remaining = size % Vec<T, ISA>::width;
      size -= remaining;

      auto vec_accu = Vec<T, ISA>::load(init);
      for (dim_t i = 0; i < size; i += Vec<T, ISA>::width) {
        auto v = Vec<T, ISA>::load(x + i);
        vec_accu = vec_reduce_func(vec_accu, vec_map_func(v));
      }

      auto accu = vec_horizontal_reduce_func(vec_accu);
      if (remaining != 0) {
        for (dim_t i = size; i < size + remaining; ++i)
          accu = scalar_reduce_func(accu, scalar_map_func(x[i]));
      }

      return accu;
    }

    template <CpuIsa ISA,
              typename T,
              typename VecReduceFunc,
              typename VecHorizontalReduceFunc,
              typename ScalarReduceFunc>
    static T vectorized_reduce_all(const T* x,
                                   dim_t size,
                                   T init,
                                   const VecReduceFunc& vec_reduce_func,
                                   const VecHorizontalReduceFunc& vec_horizontal_reduce_func,
                                   const ScalarReduceFunc& scalar_reduce_func) {
      return vectorized_map_reduce_all<ISA>(x,
                                            size,
                                            init,
                                            identity(),
                                            vec_reduce_func,
                                            vec_horizontal_reduce_func,
                                            identity(),
                                            scalar_reduce_func);
    }

    template <CpuIsa ISA, typename T>
    void rcp(const T* x, T* y, dim_t size) {
      vectorized_unary_transform<ISA>(x, y, size, Vec<T, ISA>::rcp);
    }

    template<>
    void exp<TARGET_ISA>(const float* x, float* y, dim_t size) {
      vectorized_unary_transform<TARGET_ISA>(x, y, size, Vec<float, TARGET_ISA>::exp);
    }

    template<>
    void log<TARGET_ISA>(const float* x, float* y, dim_t size) {
      vectorized_unary_transform<TARGET_ISA>(x, y, size, Vec<float, TARGET_ISA>::log);
    }

    template<>
    void sin<TARGET_ISA>(const float* x, float* y, dim_t size) {
      vectorized_unary_transform<TARGET_ISA>(x, y, size, Vec<float, TARGET_ISA>::sin);
    }

    template<>
    void cos<TARGET_ISA>(const float* x, float* y, dim_t size) {
      vectorized_unary_transform<TARGET_ISA>(x, y, size, Vec<float, TARGET_ISA>::cos);
    }

    template<>
    void tanh<TARGET_ISA>(const float* x, float* y, dim_t size) {
      vectorized_unary_transform<TARGET_ISA>(x, y, size, Vec<float, TARGET_ISA>::tanh);
    }

    template<>
    void gelu<TARGET_ISA>(const float* x, float* y, dim_t size) {
      using VecType = Vec<float, TARGET_ISA>;
      vectorized_unary_transform<TARGET_ISA>(
        x, y, size,
        [](vec_type<float, TARGET_ISA> v) {
          auto u = VecType::mul(VecType::load(0.7071067811865475f), v);
          u = VecType::add(VecType::load(1.f), VecType::erf(u));
          u = VecType::mul(v, u);
          u = VecType::mul(VecType::load(0.5f), u);
          return u;
        });
    }

    template<>
    void gelu_tanh<TARGET_ISA>(const float* x, float* y, dim_t size) {
      using VecType = Vec<float, TARGET_ISA>;
      vectorized_unary_transform<TARGET_ISA>(
        x, y, size,
        [](vec_type<float, TARGET_ISA> v) {
          auto u = VecType::mul(VecType::mul(v, v), v);
          u = VecType::mul_add(VecType::load(0.044715f), u, v);
          u = VecType::mul(VecType::load(0.7978845608028654f), u);
          u = VecType::tanh(u);
          u = VecType::add(VecType::load(1.f), u);
          u = VecType::mul(v, u);
          u = VecType::mul(VecType::load(0.5f), u);
          return u;
        });
    }

    template<>
    void gelu_sigmoid<TARGET_ISA>(const float* x, float* y, dim_t size) {
      using VecType = Vec<float, TARGET_ISA>;
      vectorized_unary_transform<TARGET_ISA>(
        x, y, size,
        [](vec_type<float, TARGET_ISA> v) {
          return VecType::div(v, VecType::add(VecType::load(1.f),
                                              VecType::exp(VecType::mul(VecType::load(-1.702f), v))));
        });
    }

    template<>
    void swish<TARGET_ISA>(const float* x, float* y, dim_t size) {
      using VecType = Vec<float, TARGET_ISA>;
      vectorized_unary_transform<TARGET_ISA>(
        x, y, size,
        [](vec_type<float, TARGET_ISA> v) {
          return VecType::div(v, VecType::add(VecType::load(1.f), VecType::exp(VecType::neg(v))));
        });
    }

    template <CpuIsa ISA, typename T>
    void add(T a, const T* x, T* y, dim_t size) {
      auto vec_a = Vec<T, ISA>::load(a);
      vectorized_unary_transform<ISA>(x, y, size,
                                      [vec_a](vec_type<T, ISA> v) {
                                        return Vec<T, ISA>::add(v, vec_a);
                                      });
    }

    template <CpuIsa ISA, typename T>
    void add(const T* a, const T* b, T* c, dim_t size) {
      vectorized_binary_transform<ISA>(a, b, c, size, Vec<T, ISA>::add);
    }

    template <CpuIsa ISA, typename T>
    void sub(const T* a, const T* b, T* c, dim_t size) {
      vectorized_binary_transform<ISA>(a, b, c, size, Vec<T, ISA>::sub);
    }

    template <CpuIsa ISA, typename T>
    void mul(T a, const T* x, T* y, dim_t size) {
      auto vec_a = Vec<T, ISA>::load(a);
      vectorized_unary_transform<ISA>(x, y, size,
                                      [vec_a](vec_type<T, ISA> v) {
                                        return Vec<T, ISA>::mul(v, vec_a);
                                      });
    }

    template <CpuIsa ISA, typename T>
    void mul(const T* a, const T* b, T* c, dim_t size) {
      vectorized_binary_transform<ISA>(a, b, c, size, Vec<T, ISA>::mul);
    }

    template <CpuIsa ISA, typename T>
    void max(T a, const T* x, T* y, dim_t size) {
      auto vec_a = Vec<T, ISA>::load(a);
      vectorized_unary_transform<ISA>(x, y, size,
                                      [vec_a](vec_type<T, ISA> v) {
                                        return Vec<T, ISA>::max(v, vec_a);
                                      });
    }

    template <CpuIsa ISA, typename T>
    void max(const T* a, const T* b, T* c, dim_t size) {
      vectorized_binary_transform<ISA>(a, b, c, size, Vec<T, ISA>::max);
    }

    template <CpuIsa ISA, typename T>
    void min(T a, const T* x, T* y, dim_t size) {
      auto vec_a = Vec<T, ISA>::load(a);
      vectorized_unary_transform<ISA>(x, y, size,
                                      [vec_a](vec_type<T, ISA> v) {
                                        return Vec<T, ISA>::min(v, vec_a);
                                      });
    }

    template <CpuIsa ISA, typename T>
    void min(const T* a, const T* b, T* c, dim_t size) {
      vectorized_binary_transform<ISA>(a, b, c, size, Vec<T, ISA>::min);
    }

    template <CpuIsa ISA, typename T>
    T reduce_sum(const T* x, dim_t size) {
      return vectorized_reduce_all<ISA>(x,
                                        size,
                                        static_cast<T>(0),
                                        Vec<T, ISA>::add,
                                        Vec<T, ISA>::reduce_add,
                                        Vec<T>::add);
    }

    template <CpuIsa ISA, typename T>
    T reduce_max(const T* x, dim_t size) {
      return vectorized_reduce_all<ISA>(x,
                                        size,
                                        std::numeric_limits<T>::lowest(),
                                        Vec<T, ISA>::max,
                                        Vec<T, ISA>::reduce_max,
                                        Vec<T>::max);
    }

    template <CpuIsa ISA, typename T>
    T reduce_amax(const T* x, dim_t size) {
      return vectorized_map_reduce_all<ISA>(x,
                                            size,
                                            static_cast<T>(0),
                                            Vec<T, ISA>::abs,
                                            Vec<T, ISA>::max,
                                            Vec<T, ISA>::reduce_max,
                                            Vec<T>::abs,
                                            Vec<T>::max);
    }

#define DECLARE_IMPL(T)                                                 \
    template void rcp<TARGET_ISA>(const T* x, T* y, dim_t size);        \
    template void add<TARGET_ISA>(T a, const T* x, T* y, dim_t size);   \
    template void add<TARGET_ISA>(const T* a, const T* b, T* c, dim_t size); \
    template void sub<TARGET_ISA>(const T* a, const T* b, T* c, dim_t size); \
    template void mul<TARGET_ISA>(T a, const T* x, T* y, dim_t size);   \
    template void mul<TARGET_ISA>(const T* a, const T* b, T* c, dim_t size); \
    template void max<TARGET_ISA>(T a, const T* x, T* y, dim_t size);   \
    template void max<TARGET_ISA>(const T* a, const T* b, T* c, dim_t size); \
    template void min<TARGET_ISA>(T a, const T* x, T* y, dim_t size);   \
    template void min<TARGET_ISA>(const T* a, const T* b, T* c, dim_t size); \
    template T reduce_sum<TARGET_ISA>(const T* x, dim_t size);          \
    template T reduce_max<TARGET_ISA>(const T* x, dim_t size);          \
    template T reduce_amax<TARGET_ISA>(const T* x, dim_t size);

    DECLARE_ALL_TYPES(DECLARE_IMPL)


    template <>
    float reduce_logsumexp<TARGET_ISA>(const float* x, dim_t size) {
      using VecType = Vec<float, TARGET_ISA>;

      const auto x_max = reduce_max<TARGET_ISA>(x, size);
      const auto vec_x_max = VecType::load(x_max);

      const auto scalar_exp_func = [x_max](vec_type<float> v) {
        return Vec<float>::exp(Vec<float>::sub(v, x_max));
      };
      const auto vec_exp_func = [vec_x_max](vec_type<float, TARGET_ISA> v) {
        return VecType::exp(VecType::sub(v, vec_x_max));
      };

      const auto exp_sum = vectorized_map_reduce_all<TARGET_ISA>(
        x,
        size,
        static_cast<float>(0),
        vec_exp_func,
        VecType::add,
        VecType::reduce_add,
        scalar_exp_func,
        Vec<float>::add);

      return std::log(exp_sum) + x_max;
    }

    template<>
    void softmax<TARGET_ISA>(const float* input,
                             const int32_t* lengths,
                             float* output,
                             dim_t batch_size,
                             dim_t depth,
                             bool log,
                             float epsilon) {
      using VecType = Vec<float, TARGET_ISA>;

      parallel_for(0, batch_size, 1, [&](dim_t begin, dim_t end) {
        for (dim_t i = begin; i < end; ++i) {
          const dim_t offset = i * depth;
          const float* x = input + offset;
          float* y = output + offset;

          dim_t size = depth;
          if (lengths) {
            size = lengths[i];

            // Directly set 0 in output for out of range positions.
            for (dim_t j = size; j < depth; ++j) {
              y[j] = 0;
            }

            if (size == 0) {
              continue;
            }
          }

          const auto x_max = reduce_max<TARGET_ISA>(x, size);
          const auto vec_x_max = VecType::load(x_max);

          const auto scalar_exp_func = [x_max](vec_type<float> v) {
            return Vec<float>::exp(Vec<float>::sub(v, x_max));
          };
          const auto vec_exp_func = [vec_x_max](vec_type<float, TARGET_ISA> v) {
            return VecType::exp(VecType::sub(v, vec_x_max));
          };

          if (log) {
            const auto exp_sum = vectorized_map_reduce_all<TARGET_ISA>(
              x,
              size,
              static_cast<float>(0),
              vec_exp_func,
              VecType::add,
              VecType::reduce_add,
              scalar_exp_func,
              Vec<float>::add);
            add<TARGET_ISA>(-x_max - std::log(exp_sum), x, y, size);
          } else {
            vectorized_unary_transform<TARGET_ISA>(x, y, size, vec_exp_func);
            const auto exp_sum = reduce_sum<TARGET_ISA>(y, size);
            mul<TARGET_ISA>(static_cast<float>(1) / (exp_sum + epsilon), y, y, size);
          }
        }
      });
    }

    CT2_FFAST_MATH_BEGIN
    template<>
    void layer_norm<TARGET_ISA>(const float* input,
                                const float* gamma,
                                const float* beta,
                                float* output,
                                dim_t batch_size,
                                dim_t depth,
                                float epsilon) {
      parallel_for(0, batch_size, 1, [&](dim_t begin, dim_t end) {
        for (dim_t i = begin; i < end; ++i) {
          const auto offset = i * depth;
          const auto* x = input + offset;
          auto* y = output + offset;
          float mean = 0;  // sum(x)/n
          float rstd = 0;  // 1/sqrt(var(x)) where var(x) = sum((x-mean)^2)/n = sum(x^2)/n - mean^2
          for (dim_t j = 0; j < depth; ++j) {
            mean += x[j];
            rstd += x[j] * x[j];
          }
          mean /= depth;
          rstd = std::max(rstd / depth - mean * mean, 0.f);
          rstd = 1.f / std::sqrt(rstd + epsilon);
          for (dim_t j = 0; j < depth; ++j) {
            y[j] = (x[j] - mean) * rstd * gamma[j] + beta[j];
          }
        }
      });
    }

    template<>
    void rms_norm<TARGET_ISA>(const float* input,
                              const float* gamma,
                              float* output,
                              dim_t batch_size,
                              dim_t depth,
                              float epsilon) {
      parallel_for(0, batch_size, 1, [&](dim_t begin, dim_t end) {
        for (dim_t i = begin; i < end; ++i) {
          const auto offset = i * depth;
          const auto* x = input + offset;
          auto* y = output + offset;

          float sum_squares = 0;
          for (dim_t j = 0; j < depth; ++j)
            sum_squares += x[j] * x[j];

          const float inv_rms = 1.f / std::sqrt(sum_squares / depth + epsilon);

          for (dim_t j = 0; j < depth; ++j)
            y[j] = x[j] * inv_rms * gamma[j];
        }
      });
    }
    CT2_FFAST_MATH_END

    template <typename RoundFunc>
    static float quantize_s8_row(const float* x,
                                 int8_t* y,
                                 dim_t depth,
                                 bool shift_to_uint8,
                                 const RoundFunc& round_func) {
      constexpr float int8_min = std::numeric_limits<int8_t>::min();
      constexpr float int8_max = std::numeric_limits<int8_t>::max();

      const auto amax = reduce_amax<TARGET_ISA>(x, depth);
      const auto scale = (amax != 0.f ? int8_max / amax : 1.f);

      if (shift_to_uint8) {
        auto* dst = reinterpret_cast<uint8_t*>(y);
        for (dim_t j = 0; j < depth; ++j)
          dst[j] = round_func(x[j] * scale - int8_min);
      } else {
        for (dim_t j = 0; j < depth; ++j)
          y[j] = round_func(x[j] * scale);
      }

      return scale;
    }

    template <typename RoundFunc>
    static void quantize_s8_batch(const float* x,
                                  int8_t* y,
                                  float* scales,
                                  dim_t batch_size,
                                  dim_t depth,
                                  bool shift_to_uint8,
                                  const RoundFunc& round_func) {
      parallel_for(0, batch_size, 1, [&](dim_t begin, dim_t end) {
        for (dim_t i = begin; i < end; ++i) {
          const auto offset = i * depth;
          const auto* src = x + offset;
          auto* dst = y + offset;
          scales[i] = quantize_s8_row(src, dst, depth, shift_to_uint8, round_func);
        }
      });
    }

    template<>
    void quantize_s8<TARGET_ISA>(const float* x,
                                 int8_t* y,
                                 float* scales,
                                 dim_t batch_size,
                                 dim_t depth,
                                 bool shift_to_uint8,
                                 bool round_before_cast) {
      if (round_before_cast)
        quantize_s8_batch(x, y, scales, batch_size, depth, shift_to_uint8, std::nearbyintf);
      else
        quantize_s8_batch(x, y, scales, batch_size, depth, shift_to_uint8, identity());
    }

  }
}
