#ifndef SOFTMAX_LUT_H_
#define SOFTMAX_LUT_H_

#include <stdint.h>
#include <stddef.h>

#include "lut_exp.h"   // extern lut_exp, LUT_EXP_SIZE, LUT_EXP_XMIN, LUT_EXP_XMAX

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// Extern LUT (provided by model.h in the translation unit):
//   static const int16_t lut_exp[1024]; // Q15, x in [-10, 0]
// And the following macros are also defined by model.h:
//   #define LUT_EXP_SIZE   1024
//   #define LUT_EXP_XMIN   -10.0
//   #define LUT_EXP_XMAX   0.0
//   #define LUT_EXP_QBITS  15
// -----------------------------------------------------------------------------
extern const int16_t lut_exp[];

// -----------------------------------------------------------------------------
// Output format:
//   - Returns probabilities in Q15: int16_t in [0, 32767]. Per-row sum ≈ 32767.
// Numerics:
//   1) Max-subtraction for stability
//   2) Scale logits prior to exp (float or fixed-point variants)
//   3) Clamp to LUT domain [-10, 0]
//   4) exp via LUT (Q15)
//   5) Normalize: p_i = round( (e_i << 15) / sum_e )
// -----------------------------------------------------------------------------

// ---- Helpers ----------------------------------------------------------------

// Clamp float to [lo, hi]
static inline float sm__clampf(float x, float lo, float hi) {
    return (x < lo) ? lo : (x > hi ? hi : x);
}

// Map real x in [LUT_EXP_XMIN, LUT_EXP_XMAX] to LUT index [0, LUT_EXP_SIZE-1]
static inline int sm__lut_index_from_x(float x) {
    const float xmin = (float)LUT_EXP_XMIN;
    const float xmax = (float)LUT_EXP_XMAX;
    if (x <= xmin) return 0;
    if (x >= xmax) return (int)LUT_EXP_SIZE - 1;
    const float t = (x - xmin) * ((float)(LUT_EXP_SIZE - 1)) / (xmax - xmin);
    int idx = (int)(t + 0.5f);               // round to nearest
    if (idx < 0) idx = 0;
    if (idx > (int)LUT_EXP_SIZE - 1) idx = (int)LUT_EXP_SIZE - 1;
    return idx;
}

// Compute integer "bins" constant for integer-only indexing (see API 2b).
// bins_per_unit = (LUT_EXP_SIZE-1)/(xmax-xmin). xmin_bins ≈ round(xmin*bins_per_unit).
static inline void sm__bins_constants(int* __restrict xmin_bins_out, int* __restrict bins_per_unit_q0_out) {
    const double bpu = (double)(LUT_EXP_SIZE - 1) / ((double)LUT_EXP_XMAX - (double)LUT_EXP_XMIN);
    const double xb  = (double)LUT_EXP_XMIN * bpu;
    const int xmin_bins = (int)(xb >= 0.0 ? (xb + 0.5) : (xb - 0.5));
    *xmin_bins_out     = xmin_bins;          // typically -1023 for size=1024, xmin=-10
    *bins_per_unit_q0_out = (int)(bpu + 0.5); // integer approx (e.g., ~102 for 102.3)
}

// ---- API 1: float-scaled logits --------------------------------------------
// Uses a float scale (e.g., 1/sqrt(dh)) exactly like your PyTorch path.
// Only float ops are the scale + indexing; exp and normalization are quantized.
//
// logits: int32 accumulators (e.g., Q·Kᵀ); N: row length
// scale:  multiply BEFORE max-shift (usual Transformer scaling, e.g., 1/sqrt(dh))
// out_q15: Q15 probabilities (length N)
static inline void softmax_row_q15f(
    const int32_t* __restrict logits,
    size_t N,
    float scale,
    int16_t* __restrict out_q15
) {
    if (N == 0) return;

    // 1) max for stability
    int32_t m = logits[0];
    for (size_t i = 1; i < N; ++i) if (logits[i] > m) m = logits[i];

    // 2) exp via LUT (Q15), accumulate sum in 64-bit
    uint64_t sum_q15 = 0;
    for (size_t i = 0; i < N; ++i) {
        float x = ((float)(logits[i] - m)) * scale;                           // stabilized & scaled
        x = sm__clampf(x, (float)LUT_EXP_XMIN, (float)LUT_EXP_XMAX);          // clamp to LUT domain
        const int idx = sm__lut_index_from_x(x);
        int16_t e_q15 = lut_exp[idx];                                         // Q15 of exp(x)
        if (e_q15 < 0) e_q15 = 0;                                             // defensive
        out_q15[i] = e_q15;                                                   // store temp
        sum_q15   += (uint16_t)e_q15;
    }

    // 3) normalize to Q15
    if (sum_q15 == 0) {
        // degenerate: uniform
        const int16_t uni = (int16_t)((1u << 15) / (uint32_t)N);
        for (size_t i = 0; i < N; ++i) out_q15[i] = uni;
        return;
    }
    for (size_t i = 0; i < N; ++i) {
        const uint32_t num = ((uint32_t)(uint16_t)out_q15[i]) << 15;          // Q30
        const int16_t p = (int16_t)((num + (uint32_t)(sum_q15 >> 1)) / (uint32_t)sum_q15);
        out_q15[i] = p;
    }
}

// ---- API 2a: fixed-point scaled logits (Q31 multiplier) --------------------
// x ≈ ((logits - max) * mul) >> rshift    (caller chooses mul,rshift to
// approximate the real scale, e.g., 1/sqrt(dh) in "real" x-units).
// Indexing to LUT still uses float for the final mapping to [−10,0] bins.
// If you want pure-integer indexing, use API 2b below.
static inline void softmax_row_q15_q31(
    const int32_t* __restrict logits,
    size_t N,
    int32_t mul,   // e.g., scale in Q31 (or any Q-format you prefer)
    int      rshift,
    int16_t* __restrict out_q15
) {
    if (N == 0) return;

    int32_t m = logits[0];
    for (size_t i = 1; i < N; ++i) if (logits[i] > m) m = logits[i];

    uint64_t sum_q15 = 0;
    for (size_t i = 0; i < N; ++i) {
        const int32_t d = logits[i] - m;
        const int64_t prod = (int64_t)d * (int64_t)mul;                        // wide mul
        const int32_t xs = (rshift >= 0) ? (int32_t)(prod >> rshift)
                                         : (int32_t)(prod << (-rshift));
        // Interpret xs as "real x" in the same units as LUT domain.
        float x = (float)xs;
        x = sm__clampf(x, (float)LUT_EXP_XMIN, (float)LUT_EXP_XMAX);
        const int idx = sm__lut_index_from_x(x);
        int16_t e_q15 = lut_exp[idx];
        if (e_q15 < 0) e_q15 = 0;
        out_q15[i] = e_q15;
        sum_q15   += (uint16_t)e_q15;
    }

    if (sum_q15 == 0) {
        const int16_t uni = (int16_t)((1u << 15) / (uint32_t)N);
        for (size_t i = 0; i < N; ++i) out_q15[i] = uni;
        return;
    }
    for (size_t i = 0; i < N; ++i) {
        const uint32_t num = ((uint32_t)(uint16_t)out_q15[i]) << 15;
        const int16_t p = (int16_t)((num + (uint32_t)(sum_q15 >> 1)) / (uint32_t)sum_q15);
        out_q15[i] = p;
    }
}

// ---- API 2b: pure-integer indexing using "LUT bins" units ------------------
// Let B = (LUT_EXP_SIZE-1)/(LUT_EXP_XMAX - LUT_EXP_XMIN). For your LUT, B≈102.3.
// Choose mul_bins, rshift such that:
//    x_bins ≈ ((logits - max) * scale * B)  (in integer bins)
// Then LUT index = clamp( x_bins - round(LUT_EXP_XMIN * B), 0, LUT_EXP_SIZE-1 ).
//
// This avoids all float ops (both scaling and indexing are integer).
static inline void softmax_row_q15_bins(
    const int32_t* __restrict logits,
    size_t N,
    int32_t mul_bins,  // approximate (scale * B) in fixed-point (together with rshift)
    int      rshift,   // right shift after multiply (>=0)
    int16_t* __restrict out_q15
) {
    if (N == 0) return;

    int xmin_bins, bins_per_unit_q0; // bins_per_unit_q0 not used directly, but handy for calibration
    sm__bins_constants(&xmin_bins, &bins_per_unit_q0);

    int32_t m = logits[0];
    for (size_t i = 1; i < N; ++i) if (logits[i] > m) m = logits[i];

    uint64_t sum_q15 = 0;
    for (size_t i = 0; i < N; ++i) {
        const int32_t d = logits[i] - m;
        const int64_t prod = (int64_t)d * (int64_t)mul_bins;                   // fixed-point
        const int32_t x_bins = (rshift >= 0) ? (int32_t)(prod >> rshift)
                                             : (int32_t)(prod << (-rshift));
        // Convert to LUT index entirely in integer "bin" space
        int idx = x_bins - xmin_bins;                                          // shift by xmin
        if (idx < 0) idx = 0;
        if (idx > (int)LUT_EXP_SIZE - 1) idx = (int)LUT_EXP_SIZE - 1;

        int16_t e_q15 = lut_exp[idx];
        if (e_q15 < 0) e_q15 = 0;
        out_q15[i] = e_q15;
        sum_q15   += (uint16_t)e_q15;
    }

    if (sum_q15 == 0) {
        const int16_t uni = (int16_t)((1u << 15) / (uint32_t)N);
        for (size_t i = 0; i < N; ++i) out_q15[i] = uni;
        return;
    }
    for (size_t i = 0; i < N; ++i) {
        const uint32_t num = ((uint32_t)(uint16_t)out_q15[i]) << 15;
        const int16_t p = (int16_t)((num + (uint32_t)(sum_q15 >> 1)) / (uint32_t)sum_q15);
        out_q15[i] = p;
    }
}

// ---- Optional: simple batched wrappers (row-major [R x C]) -----------------
static inline void softmax_rows_q15f(
    const int32_t* __restrict logits, size_t R, size_t C,
    float scale,
    int16_t* __restrict out_q15
) {
    for (size_t r = 0; r < R; ++r)
        softmax_row_q15f(&logits[r*C], C, scale, &out_q15[r*C]);
}

static inline void softmax_rows_q15_bins(
    const int32_t* __restrict logits, size_t R, size_t C,
    int32_t mul_bins, int rshift,
    int16_t* __restrict out_q15
) {
    for (size_t r = 0; r < R; ++r)
        softmax_row_q15_bins(&logits[r*C], C, mul_bins, rshift, &out_q15[r*C]);
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SOFTMAX_LUT_H_
