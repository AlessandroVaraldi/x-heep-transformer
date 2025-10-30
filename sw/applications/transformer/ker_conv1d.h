#ifndef CONV1D_H_
#define CONV1D_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---- Helpers  ----
static inline int conv1d_out_len(int Tin, int K, int stride, int pad, int dil) {
    const int eff = (K - 1) * dil + 1;
    const int num = Tin + 2*pad - eff;
    return (num >= 0) ? (num / stride + 1) : 0;
}

#ifndef SATURATE_INT8
#define SATURATE_INT8(x)  ((int8_t)((x) < -128 ? -128 : ((x) > 127 ? 127 : (x))))
#endif

// =====================================================================
// Stage A (ACCEL): grouped core — int8 x int8 -> int32 accum only
// Layout:
//   x   : [Cin, Tin]
//   w   : [Cout, Cin_g, K]   (Cin_g = Cin/groups; Cout_g = Cout/groups)
//   Y32 : [Cout, Tout]
// =====================================================================
static inline void conv1d_core(
    const int8_t* __restrict x, int Tin,
    const int8_t* __restrict w, const int32_t* __restrict b,
    int cin0, int oc0,
    int Cin_g, int Cout_g, int K, int stride, int pad, int dil,
    int Tout, int32_t* __restrict Y32
) {

    for (int oc_g = 0; oc_g < Cout_g; ++oc_g) {
        const int oc      = oc0 + oc_g;
        const int w_base  = oc * (Cin_g * K);
        const int32_t bias = b ? b[oc] : 0;
        
        for (int to = 0; to < Tout; ++to) {
            const int ti0 = to * stride - pad;
            int32_t acc = bias;

            for (int ci_l = 0; ci_l < Cin_g; ++ci_l) {
                const int ci   = cin0 + ci_l;
                const int x_ch = ci * Tin;
                const int w_ci = w_base + ci_l * K;

                for (int k = 0; k < K; ++k) {
                    const int ti = ti0 + k * dil;

                    if ((unsigned)ti < (unsigned)Tin) {
                        acc += (int32_t)x[x_ch + ti] * (int32_t)w[w_ci + k];
                    }
                }
            }

            Y32[oc * Tout + to] = acc;
        }
    }
}

static inline void conv1d_generic(
    const int8_t* __restrict x, int Tin, int Cin,
    const int8_t* __restrict w, const int32_t* __restrict b,
    int Cout, int K, int stride, int pad, int dil,
    int groups,
    int Cin_g,
    int Cout_g,
    int32_t* __restrict Y32, int Tout
) {
    // sanity
    if (Tin <= 0 || Cin <= 0 || Cout <= 0 || K <= 0 || stride <= 0 || dil <= 0 || Tout <= 0) return;
    if (groups <= 0 || Cin_g <= 0 || Cout_g <= 0) return;
    if (groups * Cin_g != Cin) return;
    if (groups * Cout_g != Cout) return;

    for (int g = 0; g < groups; ++g) {
        const int cin0   = g * Cin_g;
        const int oc0    = g * Cout_g;

        conv1d_core(
            x, Tin,
            w, b,
            cin0, oc0,
            Cin_g, Cout_g, K, stride, pad, dil,
            Tout, Y32
        );
    }
}

// =====================================================================
// Stage B (CPU): per-channel requantization int32 -> int8
// =====================================================================
static inline void requant(
    const int32_t* __restrict Y32, int Cout, int Tout,
    const int32_t* __restrict M, const int32_t* __restrict R,
    int8_t* __restrict y8
) {
    for (int oc = 0; oc < Cout; ++oc) {
        const int32_t m = M[oc];
        const int32_t r = R[oc];
        const int64_t rnd = (r > 0) ? ((int64_t)1 << (r - 1)) : 0;
        const int base = oc * Tout;
        for (int t = 0; t < Tout; ++t) {
            const int64_t prod = (int64_t)Y32[base + t] * (int64_t)m;
            int32_t rq;
            if (r > 0) rq = (int32_t)((prod + rnd) >> r);
            else if (r < 0) rq = (int32_t)(prod << (-r));
            else rq = (int32_t)prod;
            y8[base + t] = SATURATE_INT8(rq);
        }
    }
}

// ---- Public API: standard Conv1d (groups=1), w:[Cout, Cin, K] ----
static inline void conv1d(
    const int8_t* x, int Tin, int Cin,
    const int8_t* w, const int32_t* b,
    int Cout, int K, int stride, int pad, int dil,
    const int32_t* M, const int32_t* R,
    int8_t* y, int Tout, int32_t* Y32
){
    const int groups = 1;
    const int Cin_g  = Cin;
    const int Cout_g = Cout;

    conv1d_generic(
        x, Tin, Cin,
        w, b,
        Cout, K, stride, pad, dil,
        groups, Cin_g, Cout_g,
        Y32, Tout
    );
    requant(Y32, Cout, Tout, M, R, y);
}

// ---- Public API: depthwise Conv1d (groups=Cin, Cout=Cin), w:[Cin,1,K] ----
static inline void dwconv1d(
    const int8_t* x, int Tin, int Cin,
    const int8_t* w, const int32_t* b,
    int K, int stride, int pad, int dil,
    const int32_t* M, const int32_t* R,
    int8_t* y, int Tout, int32_t* Y32
){
    const int groups = Cin;
    const int Cin_g  = 1;
    const int Cout_g = 1;
    const int Cout   = Cin;

    conv1d_generic(
        x, Tin, Cin,
        w, b,
        Cout, K, stride, pad, dil,
        groups, Cin_g, Cout_g,
        Y32, Tout
    );
    requant(Y32, Cout, Tout, M, R, y);
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CONV1D_H_
