#ifndef LAYERNORM_H_
#define LAYERNORM_H_

#include <stdint.h>
#include <stddef.h>
#include <math.h>   // rsqrtf (or 1.0f/sqrtf)

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// LayerNorm (float compute, int8 I/O), over features axis C, per token t.
// Input  x: int8  [C, T]  (channels-major, contiguous)
// Output y: int8  [C, T]
// Dequant: x_f = scale_x * x
// Normalize per token t: mean/var over c in [0..C)
// Affine: y_f = ((x_f - mean) * inv_std) * gamma[c] + beta[c]   (gamma/beta optional)
// Requant: y_q = round(y_f / scale_y), clamped to int8.
// Notes:
//  - If gamma==NULL, use gamma[c]=1.
//  - If beta==NULL,  use beta[c]=0.
//  - eps should be a small positive constant, e.g., 1e-5f.
// -----------------------------------------------------------------------------

static inline int8_t ln__saturate_i8(int32_t v) {
    if (v > 127)  return 127;
    if (v < -128) return -128;
    return (int8_t)v;
}

// Single token (column t) over C features.
// x_t points to the first element of token t with stride T in the outer caller;
// this helper expects a *gather by stride* pattern, so prefer using the batched
// function below unless you already materialized a [C] vector.
static inline void layernorm_token_f32_i8o(
    const int8_t* __restrict x, // [C], dequantized via scale_x
    int C,
    float scale_x,
    const float* __restrict gamma, // [C] or NULL
    const float* __restrict beta,  // [C] or NULL
    float eps,
    float scale_y,
    int8_t* __restrict y           // [C]
) {
    // Welford’s algorithm for numerical stability
    float mean = 0.0f;
    float m2   = 0.0f;
    int   n    = 0;

    for (int c = 0; c < C; ++c) {
        const float xc = scale_x * (float)x[c];
        n++;
        const float delta  = xc - mean;
        mean += delta / (float)n;
        const float delta2 = xc - mean;
        m2   += delta2 * delta;
    }
    const float var = (n > 0) ? (m2 / (float)n) : 0.0f;
    const float inv_std = 1.0f / sqrtf(var + eps); // rsqrtf can be used if available

    const float inv_scale_y = (scale_y != 0.0f) ? (1.0f / scale_y) : 0.0f;

    if (gamma && beta) {
        for (int c = 0; c < C; ++c) {
            const float xc   = scale_x * (float)x[c];
            const float norm = (xc - mean) * inv_std;
            const float yf   = norm * gamma[c] + beta[c];
            const int32_t q  = (int32_t)lrintf(yf * inv_scale_y);
            y[c] = ln__saturate_i8(q);
        }
    } else if (gamma) { // beta = 0
        for (int c = 0; c < C; ++c) {
            const float xc   = scale_x * (float)x[c];
            const float norm = (xc - mean) * inv_std;
            const float yf   = norm * gamma[c];
            const int32_t q  = (int32_t)lrintf(yf * inv_scale_y);
            y[c] = ln__saturate_i8(q);
        }
    } else if (beta) { // gamma = 1
        for (int c = 0; c < C; ++c) {
            const float xc   = scale_x * (float)x[c];
            const float norm = (xc - mean) * inv_std;
            const float yf   = norm + beta[c];
            const int32_t q  = (int32_t)lrintf(yf * inv_scale_y);
            y[c] = ln__saturate_i8(q);
        }
    } else { // gamma = 1, beta = 0
        for (int c = 0; c < C; ++c) {
            const float xc   = scale_x * (float)x[c];
            const float norm = (xc - mean) * inv_std;
            const int32_t q  = (int32_t)lrintf(norm * inv_scale_y);
            y[c] = ln__saturate_i8(q);
        }
    }
}

// Batched over T tokens for a contiguous [C, T] layout.
static inline void layernorm_forward_f32_i8io(
    const int8_t* __restrict x,   // [C, T]
    int C,
    int T,
    float scale_x,                // dequant of x
    const float* __restrict gamma,// [C] or NULL
    const float* __restrict beta, // [C] or NULL
    float eps,
    float scale_y,                // requant of y
    int8_t* __restrict y          // [C, T]
) {
    // Process each token t independently; memory layout is [C,T]
    for (int t = 0; t < T; ++t) {
        // Compute mean/var over c for column t without materializing a [C] temp
        float mean = 0.0f;
        float m2   = 0.0f;
        int   n    = 0;

        // First pass (mean/var)
        for (int c = 0; c < C; ++c) {
            const int idx = c * T + t;                // [C,T] column-major by C
            const float xc = scale_x * (float)x[idx];
            n++;
            const float delta  = xc - mean;
            mean += delta / (float)n;
            const float delta2 = xc - mean;
            m2   += delta2 * delta;
        }
        const float var = (n > 0) ? (m2 / (float)n) : 0.0f;
        const float inv_std = 1.0f / sqrtf(var + eps);
        const float inv_scale_y = (scale_y != 0.0f) ? (1.0f / scale_y) : 0.0f;

        // Second pass (normalize + affine + requant)
        if (gamma && beta) {
            for (int c = 0; c < C; ++c) {
                const int idx = c * T + t;
                const float xc   = scale_x * (float)x[idx];
                const float norm = (xc - mean) * inv_std;
                const float yf   = norm * gamma[c] + beta[c];
                const int32_t q  = (int32_t)lrintf(yf * inv_scale_y);
                y[idx] = ln__saturate_i8(q);
            }
        } else if (gamma) {
            for (int c = 0; c < C; ++c) {
                const int idx = c * T + t;
                const float xc   = scale_x * (float)x[idx];
                const float norm = (xc - mean) * inv_std;
                const float yf   = norm * gamma[c];
                const int32_t q  = (int32_t)lrintf(yf * inv_scale_y);
                y[idx] = ln__saturate_i8(q);
            }
        } else if (beta) {
            for (int c = 0; c < C; ++c) {
                const int idx = c * T + t;
                const float xc   = scale_x * (float)x[idx];
                const float norm = (xc - mean) * inv_std;
                const float yf   = norm + beta[c];
                const int32_t q  = (int32_t)lrintf(yf * inv_scale_y);
                y[idx] = ln__saturate_i8(q);
            }
        } else {
            for (int c = 0; c < C; ++c) {
                const int idx = c * T + t;
                const float xc   = scale_x * (float)x[idx];
                const float norm = (xc - mean) * inv_std;
                const int32_t q  = (int32_t)lrintf(norm * inv_scale_y);
                y[idx] = ln__saturate_i8(q);
            }
        }
    }
}

// Convenience wrapper: no affine (gamma=1, beta=0)
static inline void layernorm_forward_f32_i8io_noaffine(
    const int8_t* __restrict x, int C, int T,
    float scale_x, float eps, float scale_y,
    int8_t* __restrict y
) {
    layernorm_forward_f32_i8io(x, C, T, scale_x, NULL, NULL, eps, scale_y, y);
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // LAYERNORM_H_
