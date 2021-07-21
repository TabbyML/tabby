#include "cpu/kernels.h"

#if defined(__AVX2__)
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

    struct identity {
      template <typename T>
      constexpr T&& operator()(T&& v) const noexcept {
        return std::forward<T>(v);
      }
    };

    template <CpuIsa ISA,
              typename T,
              typename VecMapFunc,
              typename VecReduceFunc,
              typename ScalarMapFunc,
              typename ScalarReduceFunc>
    static T vectorized_map_reduce_all(const T* x,
                                       dim_t size,
                                       T init,
                                       const VecMapFunc& vec_map_func,
                                       const VecReduceFunc& vec_reduce_func,
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
        auto v = Vec<T, ISA>::load(x + i, Vec<T, ISA>::width, init);
        vec_accu = vec_reduce_func(vec_accu, vec_map_func(v));
      }

      T values[Vec<T, ISA>::width];
      Vec<T, ISA>::store(vec_accu, values);
      const auto accu =  vectorized_map_reduce_all<ISA>(values,
                                                        Vec<T, ISA>::width,
                                                        init,
                                                        identity(),
                                                        vec_reduce_func,
                                                        identity(),
                                                        scalar_reduce_func);

      return  vectorized_map_reduce_all<ISA>(x + size,
                                             remaining,
                                             accu,
                                             vec_map_func,
                                             vec_reduce_func,
                                             scalar_map_func,
                                             scalar_reduce_func);
    }

    template <CpuIsa ISA, typename T, typename VecReduceFunc, typename ScalarReduceFunc>
    static T vectorized_reduce_all(const T* x,
                                   dim_t size,
                                   T init,
                                   const VecReduceFunc& vec_reduce_func,
                                   const ScalarReduceFunc& scalar_reduce_func) {
      return vectorized_map_reduce_all<ISA>(x,
                                            size,
                                            init,
                                            identity(),
                                            vec_reduce_func,
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
                                        Vec<T>::add);
    }

    template <CpuIsa ISA, typename T>
    T reduce_max(const T* x, dim_t size) {
      return vectorized_reduce_all<ISA>(x,
                                        size,
                                        x[0],
                                        Vec<T, ISA>::max,
                                        Vec<T>::max);
    }

    template <CpuIsa ISA, typename T>
    T reduce_amax(const T* x, dim_t size) {
      return vectorized_map_reduce_all<ISA>(x,
                                            size,
                                            static_cast<T>(0),
                                            Vec<T, ISA>::abs,
                                            Vec<T, ISA>::max,
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


    template<>
    void softmax<TARGET_ISA>(const float* input,
                             const int32_t* lengths,
                             float* output,
                             dim_t batch_size,
                             dim_t depth,
                             bool log,
                             float epsilon) {
      using VecType = Vec<float, TARGET_ISA>;

      #pragma omp parallel for
      for (dim_t i = 0; i < batch_size; ++i) {
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
            scalar_exp_func,
            Vec<float>::add);
          add<TARGET_ISA>(-x_max - std::log(exp_sum), x, y, size);
        } else {
          vectorized_unary_transform<TARGET_ISA>(x, y, size, vec_exp_func);
          const auto exp_sum = reduce_sum<TARGET_ISA>(y, size);
          mul<TARGET_ISA>(static_cast<float>(1) / (exp_sum + epsilon), y, y, size);
        }
      }
    }

  }
}
