#ifndef POSMIX_H_
#define POSMIX_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "ker_conv1d.h"

// =====================================================================
// PositionalMixing = depthwise Conv1d + residual
// y = x + DWConv(x), int8 → int32 → requant(int8) + saturated add.
// Assumption: the output scale of the DWConv matches that of x.
// Layout: x,y:[C,T]; w:[C,1,K] (packed as [C,K]); b:[C]; M,R:[C].
// Required scratch: Y32:[C,Tout] (int32).
// =====================================================================
static inline void posmix_depthwise_residual_generic(
    const int8_t* __restrict x,   // [C, T]
    int C, int T,
    const int8_t* __restrict w,   // [C, 1, K] (packed as [C,K])
    const int32_t* __restrict b,  // [C]
    const int32_t* __restrict M,  // [C]
    const int32_t* __restrict R,  // [C]
    int K, int stride, int pad, int dil,
    int8_t* __restrict y,         // [C, T]
    int32_t* __restrict Y32       // [C, Tout]
){
    if (!x || !w || !M || !R || !y || !Y32) return;
    if (C <= 0 || T <= 0 || K <= 0 || stride <= 0 || dil <= 0) return;

    const int Tout = conv1d_out_len(T, K, stride, pad, dil);
    if (Tout <= 0) return;

    // Depthwise: groups=C, Cin_g=1, Cout_g=1
    const int groups = C, Cin_g = 1, Cout_g = 1;

    // A) conv int32
    conv1d_generic(
        x, T, C,
        w, b,
        C, K, stride, pad, dil,
        groups, Cin_g, Cout_g,
        Y32, Tout
    );

    // B) per-channel requant → y (temporary: y_dw)
    requant(Y32, C, Tout, M, R, y);

    // C) residual add with saturation
    const int Tout_eff = (Tout < T) ? Tout : T;
    for (int c = 0; c < C; ++c) {
        const int base  = c * T;
        const int baseo = c * Tout;
        for (int t = 0; t < Tout_eff; ++t) {
            const int sum = (int)x[base + t] + (int)y[baseo + t];
            y[base + t] = SATURATE_INT8(sum);
        }
        // If the conv changes length (pathological case), pass-through the tail.
        for (int t = Tout_eff; t < T; ++t) {
            y[base + t] = x[base + t];
        }
    }
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // POSMIX_H_
