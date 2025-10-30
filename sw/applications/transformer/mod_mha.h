#ifndef MHA_H_
#define MHA_H_

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#include "ker_linear.h"     // linear / linear_batch
#include "ker_gemm.h"       // gemm_int32 (if you want to swap in GEMM paths later)
#include "ker_softmax.h"    // softmax_row_q15f (or bins/q31 variants)
#include "ker_layernorm.h"  // layernorm_forward_f32_i8io

#ifdef __cplusplus
extern "C" {
#endif

// --------------------------------------------------------------------------------------
// Multi-Head Self-Attention (pre-LN, int8 I/O).
//
// Data layout (channel-major, contiguous per channel):
//   x, y, work buffers are shaped as [C, T], with C=d_model.
//
// What this does (per block):
//   1) y_ln = LayerNorm(x)                         // float compute, int8 I/O
//   2) [Q|K|V] = Linear(y_ln)                      // int8 -> int8 (per-channel requant)
//   3) scores = Q·K^T (int32), scale, softmax -> P // P is Q15
//   4) ctx = P·V                                   // (Q15 × int8) -> int8 (>>15)
//   5) out = Linear(ctx)                           // int8 -> int8
//   6) y = out + x (residual, sat-int8)
//
// Windowed attention:
//   If window_radius > 0, for each t attend only to tokens j in [t-w, t+w] (clamped at edges).
//
// Notes:
//   - We keep everything channel-major to match the rest of your codebase.
//   - Softmax scaling must include (s_q * s_k) * (1/sqrt(dh)) so logits map into [-10, 0]
//     (or appropriate LUT range) before exponentiation.
// --------------------------------------------------------------------------------------

// Lightweight helpers
#ifndef LIKELY
#  define LIKELY(x)   __builtin_expect(!!(x), 1)
#endif
#ifndef UNLIKELY
#  define UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif
#ifndef SATURATE_INT8
#  define SATURATE_INT8(x)  ((int8_t)((x) < -128 ? -128 : ((x) > 127 ? 127 : (x))))
#endif

// Gather token t from [C,T] channel-major into a dense row [C]
static inline void mha__gather_token_ct(const int8_t* __restrict X, int C, int T, int t,
                                        int8_t* __restrict row_out) {
    const int tt = (t < 0) ? 0 : (t >= T ? (T-1) : t);
    for (int c = 0; c < C; ++c) {
        row_out[c] = X[c*T + tt];
    }
}

// Scatter token t (clamped) from a dense row [C] back to [C,T]
static inline void mha__scatter_token_ct(const int8_t* __restrict row_in, int C, int T, int t,
                                         int8_t* __restrict X) {
    const int tt = (t < 0) ? 0 : (t >= T ? (T-1) : t);
    for (int c = 0; c < C; ++c) {
        X[c*T + tt] = row_in[c];
    }
}

// Split concatenated QKV [3*C,T] into pointers (views) without copies.
static inline void mha__split_qkv_views(int8_t* qkv, int C, int T,
                                        int8_t** Q, int8_t** K, int8_t** V) {
    *Q = qkv;             // [C, T]
    *K = qkv + (size_t)C*T;
    *V = qkv + (size_t)2*C*T;
}

// Compute dot product between two int8 vectors of length L → int32
static inline int32_t mha__dot_i8_i8(const int8_t* __restrict a,
                                     const int8_t* __restrict b,
                                     int L) {
    int32_t s = 0;
    for (int i = 0; i < L; ++i) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
}

// Apply softmax on a vector of int32 logits with given float scale, output Q15
static inline void mha__softmax_row_q15f(const int32_t* __restrict logits,
                                         int N, float scale,
                                         int16_t* __restrict out_q15) {
    softmax_row_q15f(logits, (size_t)N, scale, out_q15);
}

// ctx_row[dh] = sum_j P_q15[j] * V_j[dh] >> 15   (int8 output with saturation)
static inline void mha__ctx_from_probs_q15(
    const int16_t* __restrict P_q15, // [J] probabilities for this row/window (sum≈1<<15)
    const int8_t*  __restrict Vblk,  // [J, dh], row-major contiguous (J rows)
    int J, int dh,
    int8_t* __restrict ctx_row       // [dh]
){
    for (int d = 0; d < dh; ++d) {
        int32_t acc = 0;
        const int8_t* vcol = &Vblk[d]; // walk down rows with stride dh
        for (int j = 0; j < J; ++j) {
            acc += (int32_t)P_q15[j] * (int32_t)(*vcol);
            vcol += dh;
        }
        // >>15 turns weighted sum back to int8 domain (since P sums to 1<<15)
        acc >>= 15;
        if (acc > 127) acc = 127;
        if (acc < -128) acc = -128;
        ctx_row[d] = (int8_t)acc;
    }
}

// --------------------------------------------------------------------------------------
// Public API
// --------------------------------------------------------------------------------------
//
// Inputs:
//   x           : [C, T] int8  (channel-major)
//   C, T        : d_model and sequence length
//   nhead       : number of heads (C % nhead == 0)
//   window_radius : 0 for global attention; otherwise attend to [t-w, t+w]
//   ln_scale_x  : dequant scale for LN input (see ker_layernorm.h)
//   ln_scale_y  : requant scale for LN output (often = ln_scale_x)
//   ln_eps      : epsilon for LN
//
// QKV projection (single fused FC producing 3*C channels):
//   fc0_W : [3*C, C]  int8 row-major
//   fc0_b : [3*C]     int32   (nullable)
//   fc0_M : [3*C]     int32   per-out-channel requant multipliers
//   fc0_R : [3*C]     int32   per-out-channel right-shifts (>=0)
//
// Output projection (C -> C):
//   fc1_W, fc1_b, fc1_M, fc1_R analogous (sizes [C,C], [C], [C], [C])
//
// Softmax scaling:
//   softmax_scale : must incorporate (s_q * s_k) * (1 / sqrt(dh))
//                   so that int32 logits * softmax_scale map to the LUT input domain.
//
// Work buffers (caller-provided):
//   Y32_scores : int32  [T * T] (global) OR [T * (2*w+1)] (windowed). Per-head reused.
//   buf_qkv    : int8   [3*C * T]
//   buf_tok    : int8   [C]                 // scratch for per-token copies
//   buf_ctx    : int8   [C * T]             // holds concatenated per-head contexts
//   probs_q15  : int16  [max(T, 2*w+1)]     // temporary softmax row
//   Y32_fc     : int32  [max(3*C, C)]       // scratch accum for linear()
//
// Output:
//   y          : [C, T] int8  (residual applied: y = out + x)
//
// --------------------------------------------------------------------------------------
static inline void mha_forward_i8_preln(
    const int8_t* __restrict x,   int C, int T,
    int nhead, int window_radius,
    float ln_scale_x, float ln_scale_y, float ln_eps,
    // QKV FC
    const int8_t*  __restrict fc0_W,
    const int32_t* __restrict fc0_b,   // nullable
    const int32_t* __restrict fc0_M,
    const int32_t* __restrict fc0_R,
    // OUT FC
    const int8_t*  __restrict fc1_W,
    const int32_t* __restrict fc1_b,   // nullable
    const int32_t* __restrict fc1_M,
    const int32_t* __restrict fc1_R,
    // Softmax scaling
    float softmax_scale,
    // Work buffers
    int32_t* __restrict Y32_scores,
    int8_t*  __restrict buf_qkv,
    int8_t*  __restrict buf_tok,
    int8_t*  __restrict buf_ctx,
    int16_t* __restrict probs_q15,
    int32_t* __restrict Y32_fc,
    // Output
    int8_t*  __restrict y
){
    if (UNLIKELY(C <= 0 || T <= 0 || nhead <= 0 || (C % nhead) != 0)) return;

    const int dh = C / nhead;

    // 1) Pre-LayerNorm: y_ln ← LN(x). We can do it in-place into y (then y becomes y_ln).
    layernorm_forward_f32_i8io(
        x, C, T,
        ln_scale_x,
        /*gamma=*/NULL, /*beta=*/NULL,  // no-affine by default; pass arrays if you have them
        ln_eps,
        ln_scale_y,
        y /* y holds y_ln afterwards: [C,T] */
    );

    // 2) QKV = Linear(y_ln) token by token → buf_qkv [3*C, T]
    //    We'll reuse linear() with per-token gathers to avoid building [T,C] batches.
    for (int t = 0; t < T; ++t) {
        mha__gather_token_ct(y, C, T, t, buf_tok); // buf_tok: [C]
        // Project to 3*C
        linear(/*x*/buf_tok, /*Cin*/C, /*w*/fc0_W, /*Cout*/3*C,
               /*b*/fc0_b, /*M*/fc0_M, /*R*/fc0_R,
               /*Y32*/Y32_fc,
               /*y*/ &buf_qkv[(size_t)t * (3*C)]); // NOTE: linear writes [Cout] contiguous
    }

    // Rearrange buf_qkv from [T, 3*C] row-major to [3*C, T] channel-major.
    // (linear wrote per-token rows; our kernels downstream assume [C,T].)
    // Do it in place with a tiny swap buffer (buf_tok) by blocks to keep it simple.
    // For robustness and clarity, we'll just re-pack into place head-by-head.
    // Create Q,K,V views on the final channel-major layout.
    int8_t *Q, *K, *V;
    mha__split_qkv_views(buf_qkv, C, T, &Q, &K, &V);

    // Temporary staging area to transpose row-major [T, Cpart] into channel-major [Cpart, T]
    // We'll do three passes over the [T, C] tiles: Q then K then V.
    // Source base pointers in row-major:
    const int8_t* srcQrm = &buf_qkv[0];            // rows of length 3*C, Q is the first C cols
    const int8_t* srcKrm = &buf_qkv[C];            // start offset +C (per row)
    const int8_t* srcVrm = &buf_qkv[2*C];          // start offset +2C (per row)

    // Transpose Q
    for (int c = 0; c < C; ++c) {
        for (int t = 0; t < T; ++t) {
            Q[c*T + t] = srcQrm[(size_t)t*(3*C) + c];
        }
    }
    // Transpose K
    for (int c = 0; c < C; ++c) {
        for (int t = 0; t < T; ++t) {
            K[c*T + t] = srcKrm[(size_t)t*(3*C) + c];
        }
    }
    // Transpose V
    for (int c = 0; c < C; ++c) {
        for (int t = 0; t < T; ++t) {
            V[c*T + t] = srcVrm[(size_t)t*(3*C) + c];
        }
    }

    // 3–4) Attention (global or windowed), per head.
    // We write ctx per head into buf_ctx, then it will be [C,T] channel-major.
    // Layout per head h:
    //   Qh,Kh,Vh as [dh, T] slices (contiguous per channel since parent is [C,T])
    for (int h = 0; h < nhead; ++h) {
        const int c0 = h * dh;
        int8_t*  Qh = &Q[c0*T];
        int8_t*  Kh = &K[c0*T];
        int8_t*  Vh = &V[c0*T];
        int8_t*  Ch = &buf_ctx[c0*T];  // context output for this head

        if (window_radius <= 0 || window_radius >= T) {
            // ---- Global attention ----
            // For each t, compute scores_t over j in [0..T-1], softmax → P_t, then ctx_t.
            for (int t = 0; t < T; ++t) {
                // Prepare logits row (int32) over all positions j.
                for (int j = 0; j < T; ++j) {
                    // Dot over dh: Qh[:,t] with Kh[:,j]
                    const int8_t* qcol = &Qh[t]; // advance by T to move along channel
                    const int8_t* kcol = &Kh[j];
                    int32_t s = 0;
                    for (int d = 0; d < dh; ++d) {
                        s += (int32_t)qcol[d*T] * (int32_t)kcol[d*T];
                    }
                    Y32_scores[(size_t)t*T + j] = s; // store in the row buffer region
                }

                // Softmax over T with float scale → probs_q15[0..T-1]
                mha__softmax_row_q15f(&Y32_scores[(size_t)t*T], T, softmax_scale, probs_q15);

                // Build a small V-block [T, dh] row-major: instead of materializing,
                // we stream Vh by rows via temporary access pattern.
                // We'll compute ctx_row[dh] directly:
                int8_t* ctx_row = &Ch[t]; // write column t (stride T through channels)
                // Construct a virtual view: Vblk[j, d] = Vh[d, j]
                // Implement mha__ctx_from_probs_q15 expecting row-major [J, dh] with contiguous rows.
                // To avoid real materialization, do an inner version tailored to our [dh, T] layout:
                for (int d = 0; d < dh; ++d) {
                    int32_t acc = 0;
                    const int8_t* vrow_d = &Vh[d*T]; // row over T for this d
                    for (int j = 0; j < T; ++j) {
                        acc += (int32_t)probs_q15[j] * (int32_t)vrow_d[j];
                    }
                    acc >>= 15;
                    if (acc > 127) acc = 127;
                    if (acc < -128) acc = -128;
                    // Write into Ch[d, t] which is Ch[(c0+d)*T + t], but we’re inside head slice:
                    Ch[d*T + t] = (int8_t)acc;
                }
            }
        } else {
            // ---- Windowed attention ----
            const int w = window_radius;
            const int win_len = 2*w + 1;

            for (int t = 0; t < T; ++t) {
                // Scores over clamped window j in [t-w, t+w]
                for (int jj = 0; jj < win_len; ++jj) {
                    const int j = t - w + jj;
                    const int jc = (j < 0) ? 0 : (j >= T ? (T-1) : j);

                    // Dot over dh: Qh[:,t] with Kh[:,jc]
                    const int8_t* qcol = &Qh[t];
                    const int8_t* kcol = &Kh[jc];
                    int32_t s = 0;
                    for (int d = 0; d < dh; ++d) {
                        s += (int32_t)qcol[d*T] * (int32_t)kcol[d*T];
                    }
                    Y32_scores[(size_t)t*win_len + jj] = s;
                }

                // Softmax over window
                mha__softmax_row_q15f(&Y32_scores[(size_t)t*win_len], win_len, softmax_scale, probs_q15);

                // Context over the same window: ctx[t, :] = sum_j P[j]*V[j,:]
                for (int d = 0; d < dh; ++d) {
                    int32_t acc = 0;
                    const int8_t* vrow_d = &Vh[d*T];
                    for (int jj = 0; jj < win_len; ++jj) {
                        const int j = t - w + jj;
                        const int jc = (j < 0) ? 0 : (j >= T ? (T-1) : j);
                        acc += (int32_t)probs_q15[jj] * (int32_t)vrow_d[jc];
                    }
                    acc >>= 15;
                    if (acc > 127) acc = 127;
                    if (acc < -128) acc = -128;
                    Ch[d*T + t] = (int8_t)acc;
                }
            }
        }
    } // heads

    // 5) Output projection per token: ctx [C,T] → y_proj [C,T] in-place into y (overwrite LN)
    for (int t = 0; t < T; ++t) {
        // gather ctx[:,t]
        mha__gather_token_ct(buf_ctx, C, T, t, buf_tok);
        // project
        linear(/*x*/buf_tok, /*Cin*/C, /*w*/fc1_W, /*Cout*/C,
               /*b*/fc1_b, /*M*/fc1_M, /*R*/fc1_R,
               /*Y32*/Y32_fc,
               /*y*/buf_tok);
        // residual add with original x[:,t] → y[:,t]
        for (int c = 0; c < C; ++c) {
            const int16_t s = (int16_t)buf_tok[c] + (int16_t)x[c*T + t];
            y[c*T + t] = SATURATE_INT8(s);
        }
    }
}

#ifdef __cplusplus
} // extern "C"
#endif
#endif // MHA_H_
