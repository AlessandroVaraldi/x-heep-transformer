#ifndef LINEAR_H_
#define LINEAR_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// =====================================================================
// Integer-only Linear (Fully-Connected) kernels for int8 inference.
//   - Inputs/weights int8 (symmetric, zp=0)
//   - Accumulators/bias int32
//   - Per-output-channel requantization: y = round((acc * M[oc]) / 2^R[oc])
// Layouts:
//   x   : [Cin]
//   w   : [Cout, Cin]          (row-major by output channel)
//   b   : [Cout]               (int32; already quantized with s_in*s_w[oc])
//   Y32 : [Cout]               (int32 accum workspace)
//   y   : [Cout]               (int8 output)
// Batch variant:
//   X   : [B, Cin]             (row-major, contiguous rows)
//   Y32 : [B, Cout]
//   Y   : [B, Cout]
// =====================================================================

// ---- Saturation helper ----
#ifndef SATURATE_INT8
#define SATURATE_INT8(x)  ((int8_t)((x) < -128 ? -128 : ((x) > 127 ? 127 : (x))))
#endif

// ---- Optional attributes / pragmas (toolchain-specific; safe no-ops if unknown) ----
#ifndef LIKELY
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

// =====================================================================
// Stage A (ACCEL): core accumulation — int8 x int8 -> int32
// =====================================================================
// Compute Y32[oc] = bias[oc] + sum_{i=0..Cin-1}( x[i] * w[oc*Cin + i] )
// Requirements:
//  - x, w, b not NULL (b may be NULL to mean zeros)
//  - Cout, Cin > 0
static inline void linear_core(
    const int8_t* __restrict x,   // [Cin]
    const int8_t* __restrict w,   // [Cout, Cin]
    const int32_t* __restrict b,  // [Cout] (can be NULL)
    int Cin,
    int Cout,
    int32_t* __restrict Y32       // [Cout]
){
    if (UNLIKELY(Cin <= 0 || Cout <= 0 || !x || !w || !Y32)) return;

    // Outer loop on output channels for cache-friendly row of weights
    for (int oc = 0; oc < Cout; ++oc) {
        const int w_row = oc * Cin;
        int32_t acc = b ? b[oc] : 0;

        // Dot product: x[0..Cin-1] · w[w_row..w_row+Cin-1]
        // (Consider XPULP sdotp4 with 4-way packing in a specialized version.)
        for (int i = 0; i < Cin; ++i) {
            acc += (int32_t)x[i] * (int32_t)w[w_row + i];
        }
        Y32[oc] = acc;
    }
}

// =====================================================================
// Stage B (CPU): per-channel requantization int32 -> int8
// =====================================================================
// For each output channel oc:
//   rq = round((Y32[oc] * M[oc]) / 2^R[oc]) with round-to-nearest (biased to +inf)
// Notes:
//  - This variant expects R[oc] >= 0 (exporter guarantees non-negative shifts).
static inline void linear_requant(
    const int32_t* __restrict Y32, // [Cout]
    const int32_t* __restrict M,   // [Cout]
    const int32_t* __restrict R,   // [Cout]  (R >= 0)
    int Cout,
    int8_t* __restrict y           // [Cout]
){
    if (UNLIKELY(Cout <= 0 || !Y32 || !M || !R || !y)) return;

    for (int oc = 0; oc < Cout; ++oc) {
        const int32_t m = M[oc];
        const int32_t r = R[oc];

        // Require non-negative shift (exported params satisfy this).
        // If you need to be defensive at runtime, uncomment the clamp:
        // const int32_t rr = (r < 0) ? 0 : r;
        const int64_t prod = (int64_t)Y32[oc] * (int64_t)m;
        const int64_t rnd  = (r > 0) ? ((int64_t)1 << (r - 1)) : 0;
        int32_t rq = (r > 0) ? (int32_t)((prod + rnd) >> r)
                             : (int32_t)(prod); // r == 0

        y[oc] = SATURATE_INT8(rq);
    }
}

// =====================================================================
// Public API: single-vector Linear
// =====================================================================
// Computes y = Requant( W·x + b ), where:
//  - x  : [Cin] int8
//  - W  : [Cout, Cin] int8
//  - b  : [Cout] int32 (optional; can be NULL)
//  - M,R: [Cout] requant parameters (per out-channel)
//  - Y32: [Cout] scratch (must be provided by caller)
//  - y  : [Cout] int8 output
static inline void linear(
    const int8_t* __restrict x,   int Cin,
    const int8_t* __restrict w,   int Cout,
    const int32_t* __restrict b,  // may be NULL
    const int32_t* __restrict M,  // requant multipliers per oc
    const int32_t* __restrict R,  // requant right-shifts per oc (R >= 0)
    int32_t* __restrict Y32,      // scratch accum [Cout]
    int8_t* __restrict y          // output [Cout]
){
    if (UNLIKELY(Cin <= 0 || Cout <= 0)) return;
    linear_core(x, w, b, Cin, Cout, Y32);
    linear_requant(Y32, M, R, Cout, y);
}

// =====================================================================
// Public API: batched Linear
// =====================================================================
// Computes Y = Requant( X·W^T + b ) for a mini-batch:
//  - X   : [B, Cin]  row-major
//  - W   : [Cout, Cin] row-major
//  - b   : [Cout] int32 (optional; may be NULL)
//  - M,R : [Cout] requant params (per out-channel)
//  - Y32 : [B, Cout] scratch (caller-provided)
//  - Y   : [B, Cout] int8 output
static inline void linear_batch(
    const int8_t* __restrict X,   int B, int Cin,
    const int8_t* __restrict W,   int Cout,
    const int32_t* __restrict b,  // may be NULL
    const int32_t* __restrict M,
    const int32_t* __restrict R,
    int32_t* __restrict Y32,      // [B, Cout]
    int8_t* __restrict Y          // [B, Cout]
){
    if (UNLIKELY(B <= 0 || Cin <= 0 || Cout <= 0 || !X || !W || !Y32 || !Y || !M || !R)) return;

    for (int n = 0; n < B; ++n) {
        const int x_off  = n * Cin;
        const int y32_off = n * Cout;
        const int y_off   = n * Cout;

        linear_core(&X[x_off], W, b, Cin, Cout, &Y32[y32_off]);
        linear_requant(&Y32[y32_off], M, R, Cout, &Y[y_off]);
    }
}

// =====================================================================
// Optional: tile-friendly accumulation for very large Cin
// =====================================================================
// If Cin is very large and you are concerned about int32 headroom or cache,
// you can accumulate in chunks and keep bias in Y32 before final requant.
static inline void linear_core_tiled(
    const int8_t* __restrict x,   // [Cin]
    const int8_t* __restrict w,   // [Cout, Cin]
    const int32_t* __restrict b,  // [Cout] (can be NULL)
    int Cin,
    int Cout,
    int tile,                      // tile size over Cin (e.g., 64 or 128)
    int32_t* __restrict Y32        // [Cout]
){
    if (UNLIKELY(Cin <= 0 || Cout <= 0 || tile <= 0)) return;

    // Initialize with bias
    for (int oc = 0; oc < Cout; ++oc) {
        Y32[oc] = b ? b[oc] : 0;
    }

    for (int t0 = 0; t0 < Cin; t0 += tile) {
        const int tlen = (t0 + tile <= Cin) ? tile : (Cin - t0);

        for (int oc = 0; oc < Cout; ++oc) {
            const int w_row = oc * Cin + t0;
            int32_t acc = 0;
            for (int i = 0; i < tlen; ++i) {
                acc += (int32_t)x[t0 + i] * (int32_t)w[w_row + i];
            }
            Y32[oc] += acc;
        }
    }
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // LINEAR_H_
