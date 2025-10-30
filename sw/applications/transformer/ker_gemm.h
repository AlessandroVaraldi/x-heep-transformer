#ifndef GEMM_H_
#define GEMM_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// =====================================================================
// Integer-only GEMM kernels for int8 inference.
//   C = Requant( A * B^T + bias )
// Formats:
//   - A : int8  [M, K]         (row-major; contiguous rows)
//   - B : int8  [N, K]         (row-major; each row is a column vector for C)
//         NOTE: B is stored as N rows of length K so that the dot with A's row
//               runs over contiguous memory for both A and B (good for XPULP).
//   - bias : int32 [N]         (optional; already quantized with s_a * s_b[n])
//   - Y32 : int32 [M, N]       (accumulator workspace)
//   - C   : int8  [M, N]       (requantized output)
//
// Requantization (per output column n):
//   C[m,n] = SATURATE_INT8( round( (Y32[m,n] * M[n]) / 2^R[n] ) )
// with R[n] >= 0. The pair (M,R) encodes (s_a * s_b[n]) / s_out.
//
// Accumulator headroom:
//   |Y32[m,n]| <= 127 * 127 * K + |bias[n]|.
//   If K can be very large, consider the tiled core variant below.
//
// Layout helpers:
//   idxA = m*K + k
//   idxB = n*K + k   (B is [N, K] row-major — i.e., B^T access is contiguous)
//   idxY = m*N + n
// =====================================================================

#ifndef SATURATE_INT8
#define SATURATE_INT8(x)  ((int8_t)((x) < -128 ? -128 : ((x) > 127 ? 127 : (x))))
#endif

#ifndef LIKELY
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

// =====================================================================
// Stage A (ACCEL): core accumulation — int8 x int8 -> int32
// Y32[m,n] = bias[n] + sum_{k=0..K-1} A[m,k] * B[n,k]
// =====================================================================
static inline void gemm_core(
    const int8_t*  __restrict A,    // [M, K]
    const int8_t*  __restrict B,    // [N, K]  (row-major)
    const int32_t* __restrict bias, // [N] (nullable)
    int M, int K, int N,
    int32_t* __restrict Y32         // [M, N]
){
    if (UNLIKELY(M <= 0 || K <= 0 || N <= 0 || !A || !B || !Y32)) return;

    for (int m = 0; m < M; ++m) {
        const int a_row = m * K;
        for (int n = 0; n < N; ++n) {
            const int b_row = n * K;
            int32_t acc = bias ? bias[n] : 0;

            // Dot(A[m,*], B[n,*]) over contiguous memory
            // (Consider XPULP sdotp4-friendly specialization with 4-byte packing.)
            for (int k = 0; k < K; ++k) {
                acc += (int32_t)A[a_row + k] * (int32_t)B[b_row + k];
            }

            Y32[m * N + n] = acc;
        }
    }
}

// =====================================================================
// Stage B (CPU): per-column requantization int32 -> int8
// rq = round( (Y32 * M[n]) / 2^R[n] ), with R[n] >= 0
// =====================================================================
static inline void gemm_requant(
    const int32_t* __restrict Y32,  // [M, N]
    const int32_t* __restrict Mv,   // [N]
    const int32_t* __restrict Rv,   // [N] (R >= 0)
    int M, int N,
    int8_t* __restrict C            // [M, N]
){
    if (UNLIKELY(M <= 0 || N <= 0 || !Y32 || !Mv || !Rv || !C)) return;

    for (int m = 0; m < M; ++m) {
        const int base = m * N;
        for (int n = 0; n < N; ++n) {
            const int32_t y   = Y32[base + n];
            const int32_t M_q = Mv[n];
            const int32_t R_q = Rv[n];

            const int64_t prod = (int64_t)y * (int64_t)M_q;
            const int64_t rnd  = (R_q > 0) ? ((int64_t)1 << (R_q - 1)) : 0;
            const int32_t rq   = (R_q > 0) ? (int32_t)((prod + rnd) >> R_q)
                                           : (int32_t)(prod); // R_q == 0
            C[base + n] = SATURATE_INT8(rq);
        }
    }
}

// =====================================================================
// Public API: GEMM with int8 I/O and per-column requant
// C = Requant( A * B^T + bias )
// - A : [M,K] int8
// - B : [N,K] int8
// - bias : [N] int32 (nullable)
// - Mv,Rv : [N] requant params
// - Y32 : [M,N] int32 scratch (caller-provided)
// - C : [M,N] int8 out
// =====================================================================
static inline void gemm(
    const int8_t*  __restrict A,    int M, int K,
    const int8_t*  __restrict B,    int N,
    const int32_t* __restrict bias, // [N] (nullable)
    const int32_t* __restrict Mv,   // [N]
    const int32_t* __restrict Rv,   // [N], R >= 0
    int32_t*       __restrict Y32,  // [M,N]
    int8_t*        __restrict C     // [M,N]
){
    if (UNLIKELY(M <= 0 || K <= 0 || N <= 0)) return;
    gemm_core(A, B, bias, M, K, N, Y32);
    gemm_requant(Y32, Mv, Rv, M, N, C);
}

// =====================================================================
// Variant: GEMM producing int32 output (no requant), useful if the next
//          stage remains in int32 or does FP LayerNorm.
// =====================================================================
static inline void gemm_int32(
    const int8_t*  __restrict A,    int M, int K,
    const int8_t*  __restrict B,    int N,
    const int32_t* __restrict bias, // [N] (nullable)
    int32_t*       __restrict Y32   // [M,N] out
){
    gemm_core(A, B, bias, M, K, N, Y32);
}

// =====================================================================
// Tiled core over K to reduce int32 peak and improve cache locality.
// After each K-tile we can optionally right-shift partial sums to avoid
// overflow; that shift can be absorbed into (M,R) offline.
// =====================================================================
static inline void gemm_core_tiled(
    const int8_t*  __restrict A,    // [M, K]
    const int8_t*  __restrict B,    // [N, K]
    const int32_t* __restrict bias, // [N] (nullable)
    int M, int K, int N,
    int Ktile,                       // e.g., 64 or 128
    int inter_shift,                 // >=0, applied after each tile
    int32_t* __restrict Y32          // [M, N]
){
    if (UNLIKELY(M <= 0 || K <= 0 || N <= 0 || Ktile <= 0 || !A || !B || !Y32)) return;

    // Initialize with bias once
    for (int m = 0; m < M; ++m) {
        const int base = m * N;
        for (int n = 0; n < N; ++n) {
            Y32[base + n] = bias ? bias[n] : 0;
        }
    }

    for (int k0 = 0; k0 < K; k0 += Ktile) {
        const int klen = (k0 + Ktile <= K) ? Ktile : (K - k0);

        for (int m = 0; m < M; ++m) {
            const int a_row = m * K + k0;
            for (int n = 0; n < N; ++n) {
                const int b_row = n * K + k0;
                int32_t acc = 0;
                for (int k = 0; k < klen; ++k) {
                    acc += (int32_t)A[a_row + k] * (int32_t)B[b_row + k];
                }
                if (inter_shift > 0) acc >>= inter_shift;
                Y32[m * N + n] += acc;
            }
        }
    }
}

// =====================================================================
// Notes for optimization on XPULP:
//  - Pack 4 int8 into one int32 and use sdotp4-like intrinsics to process
//    4 MACs per iteration (with proper alignment, tail handling for K%4).
//  - Unroll inner K loop (e.g., by 4 or 8).
//  - Ensure A, B, Y32 are at least 4-byte aligned; prefer 8- or 16-byte.
//  - Consider blocking in N as well if Y32 does not fit well in cache.
// =====================================================================

#ifdef __cplusplus
} // extern "C"
#endif

#endif // GEMM_H_
