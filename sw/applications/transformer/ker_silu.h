#ifndef SILU_LUT_H_
#define SILU_LUT_H_

#include <stdint.h>
#include <stddef.h>

#include "lut_sigmoid.h"   // extern lut_sigmoid, LUT_SIGMOID_SIZE

#ifdef __cplusplus
extern "C" {
#endif

// If not already defined elsewhere
#ifndef SATURATE_INT8
#define SATURATE_INT8(x)  ((int8_t)((x) < -128 ? -128 : ((x) > 127 ? 127 : (x))))
#endif

// =====================================================================
// SiLU kernel (integer-only):
//   y_q = SAT8( ( x_q * sigmoid_q15(idx(x_q)) ) >> 15 )
//
// Inputs:
//   x      : int8 activations (length N)
//   N      : number of elements
//   lut    : pointer to int16_t sigmoid LUT in Q15 (e.g., `lut_sigmoid` from model.h)
//   L      : LUT length (e.g., LUT_SIGMOID_SIZE)
//   alpha  : fixed-point slope for index mapping
//   beta   : fixed-point offset for index mapping
//   rshift : right shift for index mapping
//
// Index mapping (no interpolation):
//   idx = clamp( ((int32_t)x_q * alpha + beta) >> rshift, 0, L-1 )
//
// How to choose (alpha, beta, rshift) offline (one-time, no FP at runtime):
//   Let real x ≈ x_q * s_in, LUT domain [xmin, xmax], size L.
//   We want: idx ≈ ( (x_q * s_in - xmin) / (xmax - xmin) ) * (L-1)
//   Pick rshift (e.g., 12..16) and set:
//     scale  = (L-1) / (xmax - xmin)
//     alpha  = round( s_in * scale * 2^rshift )
//     beta   = round( (-xmin) * scale * 2^rshift )
//   Example with xmin=-8, xmax=8, L=1024, s_in≈0.0625 (Q4.4):
//     scale=(1023/16)=63.9375 → choose rshift=14
//     alpha≈ round(0.0625 * 63.9375 * 16384) = 65536
//     beta ≈ round(8 * 63.9375 * 16384)      = 8,388,608
//   Tune rshift to avoid overflow in (x_q*alpha + beta).
// =====================================================================
static inline void silu_lut_q15(
    const int8_t* __restrict x, size_t N,
    const int16_t* __restrict lut, int L,
    int32_t alpha, int32_t beta, int rshift,
    int8_t* __restrict y
) {
    if (!x || !y || !lut || L <= 0) return;

    // Precompute clamp bounds
    const int idx_max = L - 1;

    for (size_t i = 0; i < N; ++i) {
        // Integer-only LUT indexing
        const int32_t xi   = (int32_t)x[i];
        int32_t idx = (int32_t)(((int64_t)xi * (int64_t)alpha + (int64_t)beta) >> rshift);
        if (idx < 0) idx = 0;
        else if (idx > idx_max) idx = idx_max;

        // Fetch Q15 sigmoid
        const int32_t sig_q15 = (int32_t)lut[idx];  // [0 .. 32767]

        // y = x * sigmoid(x); x is int8 (Q0), sigmoid is Q15
        // Round-to-nearest before shifting down by 15.
        const int32_t prod = xi * sig_q15;          // fits in int32 (<= 127*32767)
        const int32_t rq   = (prod + (1 << 14)) >> 15;

        y[i] = SATURATE_INT8(rq);
    }
}

// =====================================================================
// Convenience wrapper bound to model.h symbols (if included):
//   - Uses `lut_sigmoid` and `LUT_SIGMOID_SIZE`.
//   - You still pass (alpha, beta, rshift) decided offline.
// =====================================================================
#ifdef LUT_SIGMOID_SIZE
static inline void silu_lut_q15_model(
    const int8_t* __restrict x, size_t N,
    int32_t alpha, int32_t beta, int rshift,
    int8_t* __restrict y
) {
    extern const int16_t lut_sigmoid[LUT_SIGMOID_SIZE];
    silu_lut_q15(x, N, lut_sigmoid, (int)LUT_SIGMOID_SIZE, alpha, beta, rshift, y);
}
#endif

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SILU_LUT_H_
