/*
 * Project : SIMD_Utils
 * Version : 0.2.5
 * Author  : JishinMaster
 * Licence : BSD-2
 */

/*
   AVX implementation of sin, cos, sincos, exp and log

   Based on "sse_mathfun.h", by Julien Pommier
   and "avx_mathfun.h" by Giovanni Garberoglio
   http://gruntthepeon.free.fr/ssemath/
*/
#include <immintrin.h>
#include <limits>

#ifdef _MSC_VER
# define ALIGN64_BEG __declspec(align(64))
# define ALIGN64_END
#else /* gcc or icc */
# define ALIGN64_BEG
# define ALIGN64_END __attribute__((aligned(64)))
#endif

typedef __m512 v16sf;
typedef __m512i v16si;

#define _PS512_CONST(Name, Val) \
    static const ALIGN64_BEG float _ps512_##Name[16] ALIGN64_END = {Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val}
#define _PI32_CONST512(Name, Val) \
    static const ALIGN64_BEG int _pi32_512_##Name[16] ALIGN64_END = {Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val}
#define _PS512_CONST_TYPE(Name, Type, Val)                              \
    static const ALIGN64_BEG Type _ps512_##Name[16] ALIGN64_END = {Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val}

_PI32_CONST512(0, 0);
_PI32_CONST512(1, 1);
_PI32_CONST512(inv1, ~1);
_PI32_CONST512(2, 2);
_PI32_CONST512(4, 4);
_PI32_CONST512(0x7f, 0x7f);

_PS512_CONST(1, 1.0f);
_PS512_CONST(0p5, 0.5f);

_PS512_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS512_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS512_CONST_TYPE(sign_mask, int, (int) 0x80000000);
_PS512_CONST_TYPE(inv_sign_mask, int, ~0x80000000);
_PS512_CONST(cephes_SQRTHF, 0.707106781186547524f);
_PS512_CONST(cephes_log_p0, 7.0376836292E-2f);
_PS512_CONST(cephes_log_p1, -1.1514610310E-1f);
_PS512_CONST(cephes_log_p2, 1.1676998740E-1f);
_PS512_CONST(cephes_log_p3, -1.2420140846E-1f);
_PS512_CONST(cephes_log_p4, +1.4249322787E-1f);
_PS512_CONST(cephes_log_p5, -1.6668057665E-1f);
_PS512_CONST(cephes_log_p6, +2.0000714765E-1f);
_PS512_CONST(cephes_log_p7, -2.4999993993E-1f);
_PS512_CONST(cephes_log_p8, +3.3333331174E-1f);
_PS512_CONST(cephes_log_q1, -2.12194440e-4f);
_PS512_CONST(cephes_log_q2, 0.693359375f);

_PS512_CONST(exp_hi, 88.3762626647949f);
_PS512_CONST(exp_lo, -88.3762626647949f);

_PS512_CONST(cephes_LOG2EF, 1.44269504088896341f);
_PS512_CONST(cephes_exp_C1, 0.693359375f);
_PS512_CONST(cephes_exp_C2, -2.12194440e-4f);

_PS512_CONST(cephes_exp_p0, 1.9875691500E-4f);
_PS512_CONST(cephes_exp_p1, 1.3981999507E-3f);
_PS512_CONST(cephes_exp_p2, 8.3334519073E-3f);
_PS512_CONST(cephes_exp_p3, 4.1665795894E-2f);
_PS512_CONST(cephes_exp_p4, 1.6666665459E-1f);
_PS512_CONST(cephes_exp_p5, 5.0000001201E-1f);

_PS512_CONST(minus_cephes_DP1, -0.78515625f);
_PS512_CONST(minus_cephes_DP2, -2.4187564849853515625e-4f);
_PS512_CONST(minus_cephes_DP3, -3.77489497744594108e-8f);
_PS512_CONST(sincof_p0, -1.9515295891E-4f);
_PS512_CONST(sincof_p1, 8.3321608736E-3f);
_PS512_CONST(sincof_p2, -1.6666654611E-1f);
_PS512_CONST(coscof_p0, 2.443315711809948E-005f);
_PS512_CONST(coscof_p1, -1.388731625493765E-003f);
_PS512_CONST(coscof_p2, 4.166664568298827E-002f);
_PS512_CONST(cephes_FOPI, 1.27323954473516f);  // 4 / M_PI

/* natural logarithm computed for 8 simultaneous float
   return -inf for x == 0
   return NaN for x < 0
*/
static inline v16sf log512_ps(v16sf x)
{
    v16si imm0;
    v16sf zero = _mm512_setzero_ps();
    v16sf one = *(v16sf *) _ps512_1;

    __mmask16 zero_mask = _mm512_cmp_ps_mask(x, zero, _CMP_EQ_OQ);
    v16sf invalid_mask = _mm512_castsi512_ps(_mm512_movm_epi32(_mm512_cmp_ps_mask(x, zero, _CMP_LE_OS)));
    x = _mm512_max_ps(x, *(v16sf *) _ps512_min_norm_pos); /* cut off denormalized stuff */

    // can be done with AVX2
    imm0 = _mm512_srli_epi32(_mm512_castps_si512(x), 23);

    /* keep only the fractional part */
    x = _mm512_and_ps(x, *(v16sf *) _ps512_inv_mant_mask);
    x = _mm512_or_ps(x, *(v16sf *) _ps512_0p5);

    // this is again another AVX2 instruction
    imm0 = _mm512_sub_epi32(imm0, *(v16si *) _pi32_512_0x7f);
    v16sf e = _mm512_cvtepi32_ps(imm0);

    e = _mm512_add_ps(e, one);

    v16sf mask = _mm512_castsi512_ps(_mm512_movm_epi32(_mm512_cmp_ps_mask(x, *(v16sf *) _ps512_cephes_SQRTHF, _CMP_LT_OS)));

    v16sf tmp = _mm512_and_ps(x, mask);
    x = _mm512_sub_ps(x, one);
    e = _mm512_sub_ps(e, _mm512_and_ps(one, mask));
    x = _mm512_add_ps(x, tmp);

    v16sf z = _mm512_mul_ps(x, x);

    v16sf y = _mm512_fmadd_ps(*(v16sf *) _ps512_cephes_log_p0, x, *(v16sf *) _ps512_cephes_log_p1);
    y = _mm512_fmadd_ps(y, x, *(v16sf *) _ps512_cephes_log_p2);
    y = _mm512_fmadd_ps(y, x, *(v16sf *) _ps512_cephes_log_p3);
    y = _mm512_fmadd_ps(y, x, *(v16sf *) _ps512_cephes_log_p4);
    y = _mm512_fmadd_ps(y, x, *(v16sf *) _ps512_cephes_log_p5);
    y = _mm512_fmadd_ps(y, x, *(v16sf *) _ps512_cephes_log_p6);
    y = _mm512_fmadd_ps(y, x, *(v16sf *) _ps512_cephes_log_p7);
    y = _mm512_fmadd_ps(y, x, *(v16sf *) _ps512_cephes_log_p8);
    y = _mm512_mul_ps(y, x);

    y = _mm512_mul_ps(y, z);

    y = _mm512_fmadd_ps(e, *(v16sf *) _ps512_cephes_log_q1, y);
    y = _mm512_fnmadd_ps(z, *(v16sf *) _ps512_0p5, y);

    tmp = _mm512_fmadd_ps(e, *(v16sf *) _ps512_cephes_log_q2, y);
    x = _mm512_add_ps(x, tmp);
    x = _mm512_or_ps(x, invalid_mask);  // negative arg will be NAN

    v16sf negative_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    x = _mm512_mask_blend_ps(zero_mask, x, negative_inf);

    return x;
}

static inline v16sf exp512_ps(v16sf x)
{
    v16sf tmp = _mm512_setzero_ps(), fx;
    v16si imm0;
    v16sf one = *(v16sf *) _ps512_1;

    x = _mm512_min_ps(x, *(v16sf *) _ps512_exp_hi);
    x = _mm512_max_ps(x, *(v16sf *) _ps512_exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = _mm512_fmadd_ps(x, *(v16sf *) _ps512_cephes_LOG2EF, *(v16sf *) _ps512_0p5);
    tmp = _mm512_floor_ps(fx);

    /* if greater, substract 1 */
    v16sf mask = _mm512_castsi512_ps(_mm512_movm_epi32(_mm512_cmp_ps_mask(tmp, fx, _CMP_GT_OS)));
    mask = _mm512_and_ps(mask, one);
    fx = _mm512_sub_ps(tmp, mask);

    x = _mm512_fnmadd_ps(fx, *(v16sf *) _ps512_cephes_exp_C1, x);
    x = _mm512_fnmadd_ps(fx, *(v16sf *) _ps512_cephes_exp_C2, x);

    v16sf z = _mm512_mul_ps(x, x);

    v16sf y = _mm512_fmadd_ps(*(v16sf *) _ps512_cephes_exp_p0, x, *(v16sf *) _ps512_cephes_exp_p1);
    y = _mm512_fmadd_ps(y, x, *(v16sf *) _ps512_cephes_exp_p2);
    y = _mm512_fmadd_ps(y, x, *(v16sf *) _ps512_cephes_exp_p3);
    y = _mm512_fmadd_ps(y, x, *(v16sf *) _ps512_cephes_exp_p4);
    y = _mm512_fmadd_ps(y, x, *(v16sf *) _ps512_cephes_exp_p5);
    y = _mm512_fmadd_ps(y, z, x);
    y = _mm512_add_ps(y, one);

    /* build 2^n */
    imm0 = _mm512_cvttps_epi32(fx);
    // another two AVX2 instructions
    imm0 = _mm512_add_epi32(imm0, *(v16si *) _pi32_512_0x7f);
    imm0 = _mm512_slli_epi32(imm0, 23);
    v16sf pow2n = _mm512_castsi512_ps(imm0);
    y = _mm512_mul_ps(y, pow2n);
    return y;
}

static inline v16sf sin512_ps(v16sf x)
{  // any x
    v16sf sign_bit, y;
    v16si imm0, imm2;

    sign_bit = x;
    /* take the absolute value */
    x = _mm512_and_ps(x, *(v16sf *) _ps512_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit = _mm512_and_ps(sign_bit, *(v16sf *) _ps512_sign_mask);

    /* scale by 4/Pi */
    y = _mm512_mul_ps(x, *(v16sf *) _ps512_cephes_FOPI);

    /* store the integer part of y in mm0 */
    imm2 = _mm512_cvttps_epi32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    // another two AVX2 instruction
    imm2 = _mm512_add_epi32(imm2, *(v16si *) _pi32_512_1);
    imm2 = _mm512_and_si512(imm2, *(v16si *) _pi32_512_inv1);
    y = _mm512_cvtepi32_ps(imm2);

    /* get the swap sign flag */
    imm0 = _mm512_and_si512(imm2, *(v16si *) _pi32_512_4);
    imm0 = _mm512_slli_epi32(imm0, 29);
    /* get the polynom selection mask
     there is one polynom for 0 <= x <= Pi/4
     and another one for Pi/4<x<=Pi/2

     Both branches will be computed.
  */
    imm2 = _mm512_and_si512(imm2, *(v16si *) _pi32_512_2);
    imm2 = (__m512i) _mm512_maskz_set1_epi32(_mm512_cmpeq_epi32_mask(imm2, *(v16si *) _pi32_512_0), -1);


    v16sf swap_sign_bit = _mm512_castsi512_ps(imm0);

#if 1
    // Cast integer 0000 FFFF (negative int) to mmask type. Is there a better way?
    __mmask16 poly_mask = _mm512_cmplt_epi32_mask(imm2, _mm512_setzero_si512());
#else
    v16sf poly_mask = _mm512_castsi512_ps(imm2);
#endif

    sign_bit = _mm512_xor_ps(sign_bit, swap_sign_bit);

    /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
    x = _mm512_fmadd_ps(y, *(v16sf *) _ps512_minus_cephes_DP1, x);
    x = _mm512_fmadd_ps(y, *(v16sf *) _ps512_minus_cephes_DP2, x);
    x = _mm512_fmadd_ps(y, *(v16sf *) _ps512_minus_cephes_DP3, x);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    v16sf z = _mm512_mul_ps(x, x);

    y = _mm512_fmadd_ps(*(v16sf *) _ps512_coscof_p0, z, *(v16sf *) _ps512_coscof_p1);
    y = _mm512_fmadd_ps(y, z, *(v16sf *) _ps512_coscof_p2);
    y = _mm512_mul_ps(y, z);
    y = _mm512_mul_ps(y, z);
    y = _mm512_fnmadd_ps(z, *(v16sf *) _ps512_0p5, y);
    y = _mm512_add_ps(y, *(v16sf *) _ps512_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    v16sf y2 = _mm512_fmadd_ps(*(v16sf *) _ps512_sincof_p0, z, *(v16sf *) _ps512_sincof_p1);
    y2 = _mm512_fmadd_ps(y2, z, *(v16sf *) _ps512_sincof_p2);
    y2 = _mm512_mul_ps(y2, z);
    y2 = _mm512_fmadd_ps(y2, x, x);

    /* select the correct result from the two polynoms */
#if 1
    y = _mm512_mask_blend_ps(poly_mask, y, y2);
#else
    y2 = _mm512_and_ps(poly_mask, y2);  //, xmm3);
    y = _mm512_andnot_ps(poly_mask, y);
    y = _mm512_add_ps(y, y2);
#endif

    /* update the sign */
    y = _mm512_xor_ps(y, sign_bit);

    return y;
}

/* almost the same as sin_ps */
static inline v16sf cos512_ps(v16sf x)
{  // any x
    v16sf y;
    v16si imm0, imm2;

    /* take the absolute value */
    x = _mm512_and_ps(x, *(v16sf *) _ps512_inv_sign_mask);

    /* scale by 4/Pi */
    y = _mm512_mul_ps(x, *(v16sf *) _ps512_cephes_FOPI);

    /* store the integer part of y in mm0 */
    imm2 = _mm512_cvttps_epi32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    imm2 = _mm512_add_epi32(imm2, *(v16si *) _pi32_512_1);
    imm2 = _mm512_and_si512(imm2, *(v16si *) _pi32_512_inv1);
    y = _mm512_cvtepi32_ps(imm2);
    imm2 = _mm512_sub_epi32(imm2, *(v16si *) _pi32_512_2);

    /* get the swap sign flag */
    imm0 = _mm512_andnot_si512(imm2, *(v16si *) _pi32_512_4);
    imm0 = _mm512_slli_epi32(imm0, 29);
    /* get the polynom selection mask */
    imm2 = _mm512_and_si512(imm2, *(v16si *) _pi32_512_2);
    imm2 = (__m512i) _mm512_maskz_set1_epi32(_mm512_cmpeq_epi32_mask(imm2, *(v16si *) _pi32_512_0), -1);

    v16sf sign_bit = _mm512_castsi512_ps(imm0);

#if 1
    // Cast integer 0000 FFFF (negative int) to mmask type. Is there a better way?
    __mmask16 poly_mask = _mm512_cmplt_epi32_mask(imm2, _mm512_setzero_si512());
#else
    v16sf poly_mask = _mm512_castsi512_ps(imm2);
#endif

    /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
    x = _mm512_fmadd_ps(y, *(v16sf *) _ps512_minus_cephes_DP1, x);
    x = _mm512_fmadd_ps(y, *(v16sf *) _ps512_minus_cephes_DP2, x);
    x = _mm512_fmadd_ps(y, *(v16sf *) _ps512_minus_cephes_DP3, x);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    v16sf z = _mm512_mul_ps(x, x);

    y = _mm512_fmadd_ps(*(v16sf *) _ps512_coscof_p0, z, *(v16sf *) _ps512_coscof_p1);
    y = _mm512_fmadd_ps(y, z, *(v16sf *) _ps512_coscof_p2);
    y = _mm512_mul_ps(y, z);
    y = _mm512_mul_ps(y, z);
    y = _mm512_fnmadd_ps(z, *(v16sf *) _ps512_0p5, y);
    y = _mm512_add_ps(y, *(v16sf *) _ps512_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    v16sf y2 = _mm512_fmadd_ps(*(v16sf *) _ps512_sincof_p0, z, *(v16sf *) _ps512_sincof_p1);
    y2 = _mm512_fmadd_ps(y2, z, *(v16sf *) _ps512_sincof_p2);
    y2 = _mm512_mul_ps(y2, z);
    y2 = _mm512_fmadd_ps(y2, x, x);

    /* select the correct result from the two polynoms */
#if 1
    y = _mm512_mask_blend_ps(poly_mask, y, y2);
#else
    y2 = _mm512_and_ps(poly_mask, y2);  //, xmm3);
    y = _mm512_andnot_ps(poly_mask, y);
    y = _mm512_add_ps(y, y2);
#endif

    /* update the sign */
    y = _mm512_xor_ps(y, sign_bit);

    return y;
}

static inline void sincos512_ps(v16sf x, v16sf *s, v16sf *c)
{
    v16sf xmm1, xmm2, sign_bit_sin, y;
    v16si imm0, imm2, imm4;

    sign_bit_sin = x;
    /* take the absolute value */
    x = _mm512_and_ps(x, *(v16sf *) _ps512_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit_sin = _mm512_and_ps(sign_bit_sin, *(v16sf *) _ps512_sign_mask);

    /* scale by 4/Pi */
    y = _mm512_mul_ps(x, *(v16sf *) _ps512_cephes_FOPI);

    /* store the integer part of y in imm2 */
    imm2 = _mm512_cvttps_epi32(y);

    /* j=(j+1) & (~1) (see the cephes sources) */
    imm2 = _mm512_add_epi32(imm2, *(v16si *) _pi32_512_1);
    imm2 = _mm512_and_si512(imm2, *(v16si *) _pi32_512_inv1);

    y = _mm512_cvtepi32_ps(imm2);
    imm4 = imm2;

    /* get the swap sign flag for the sine */
    imm0 = _mm512_and_si512(imm2, *(v16si *) _pi32_512_4);
    imm0 = _mm512_slli_epi32(imm0, 29);
    // v16sf swap_sign_bit_sin = _mm512_castsi512_ps(imm0);

    /* get the polynom selection mask for the sine*/
    imm2 = _mm512_and_si512(imm2, *(v16si *) _pi32_512_2);
    imm2 = (__m512i) _mm512_maskz_set1_epi32(_mm512_cmpeq_epi32_mask(imm2, *(v16si *) _pi32_512_0), -1);
    // v16sf poly_mask = _mm512_castsi512_ps(imm2);

    v16sf swap_sign_bit_sin = _mm512_castsi512_ps(imm0);

#if 1
    // Cast integer 0000 FFFF (negative int) to mmask type. Is there a better way?
    __mmask16 poly_mask = _mm512_cmplt_epi32_mask(imm2, _mm512_setzero_si512());
#else
    v16sf poly_mask = _mm512_castsi512_ps(imm2);
#endif

    /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
    x = _mm512_fmadd_ps(y, *(v16sf *) _ps512_minus_cephes_DP1, x);
    x = _mm512_fmadd_ps(y, *(v16sf *) _ps512_minus_cephes_DP2, x);
    x = _mm512_fmadd_ps(y, *(v16sf *) _ps512_minus_cephes_DP3, x);

    imm4 = _mm512_sub_epi32(imm4, *(v16si *) _pi32_512_2);
    imm4 = _mm512_andnot_si512(imm4, *(v16si *) _pi32_512_4);
    imm4 = _mm512_slli_epi32(imm4, 29);

    v16sf sign_bit_cos = _mm512_castsi512_ps(imm4);

    sign_bit_sin = _mm512_xor_ps(sign_bit_sin, swap_sign_bit_sin);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    v16sf z = _mm512_mul_ps(x, x);

    y = _mm512_fmadd_ps(*(v16sf *) _ps512_coscof_p0, z, *(v16sf *) _ps512_coscof_p1);
    y = _mm512_fmadd_ps(y, z, *(v16sf *) _ps512_coscof_p2);
    y = _mm512_mul_ps(y, z);
    y = _mm512_mul_ps(y, z);
    y = _mm512_fnmadd_ps(z, *(v16sf *) _ps512_0p5, y);
    y = _mm512_add_ps(y, *(v16sf *) _ps512_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */
    v16sf y2 = _mm512_fmadd_ps(*(v16sf *) _ps512_sincof_p0, z, *(v16sf *) _ps512_sincof_p1);
    y2 = _mm512_fmadd_ps(y2, z, *(v16sf *) _ps512_sincof_p2);
    y2 = _mm512_mul_ps(y2, z);
    y2 = _mm512_fmadd_ps(y2, x, x);

    /* select the correct result from the two polynoms */
#if 1
    xmm1 = _mm512_mask_blend_ps(poly_mask, y, y2);
    xmm2 = _mm512_mask_blend_ps(poly_mask, y2, y);
#else
    v16sf ysin2 = _mm512_and_ps(poly_mask, y2);
    v16sf ysin1 = _mm512_andnot_ps(poly_mask, y);
    y2 = _mm512_sub_ps(y2, ysin2);
    y = _mm512_sub_ps(y, ysin1);
    xmm1 = _mm512_add_ps(ysin1, ysin2);
    xmm2 = _mm512_add_ps(y, y2);
#endif

    /* update the sign */
    *s = _mm512_xor_ps(xmm1, sign_bit_sin);
    *c = _mm512_xor_ps(xmm2, sign_bit_cos);
}
