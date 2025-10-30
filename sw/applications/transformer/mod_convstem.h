#ifndef CONVSTEM_H_
#define CONVSTEM_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "ker_conv1d.h"
#include "ker_silu.h"

// =====================================================================
// ConvStem = Conv1d (with BN already folded in weights/bias) + SiLU
//
// Quantization:
//   - Input  : int8 (per-tensor scale s_in decided offline)
//   - Conv   : int8 x int8 -> int32 accum (+ bias_int32)
//   - Requant: per-channel (M[oc], R[oc]) -> int8 (this yields the
//              activation scale that the LUT index mapping is based on)
//   - Act    : SiLU via LUT: y = x * sigmoid(x), with sigmoid in Q15
//
// LUT index mapping (no interpolation):
//   idx = clamp( ((x_q * alpha) + beta) >> rshift, 0, L-1 )
// where (alpha, beta, rshift) are precomputed offline for the layer’s
// post-requant activation scale and the LUT domain (e.g., [-8, 8]).
// =====================================================================

// Generic API: execute ConvStem with provided tensors and LUT.
// Shapes:
//   x      : [Cin, Tin] (int8)
//   w      : [Cout, Cin, K] (int8)
//   b      : [Cout] (int32)
//   M, R   : [Cout] (int32) per-channel requant params
//   y      : [Cout, Tout] (int8)  (Tout computed by conv1d_out_len)
//   Y32    : scratch buffer [Cout, Tout] (int32)
//   lut    : int16 Q15 sigmoid (size L)
// Mapping:
//   alpha, beta, rshift: affine mapping constants for LUT indexing.
static inline void convstem_generic(
    const int8_t* __restrict x, int Tin, int Cin,
    const int8_t* __restrict w, const int32_t* __restrict b,
    int Cout, int K, int stride, int pad, int dil,
    const int32_t* __restrict M, const int32_t* __restrict R,
    const int16_t* __restrict lut, int L,
    int32_t alpha, int32_t beta, int rshift,
    int8_t* __restrict y, int32_t* __restrict Y32
) {
    
    if (!x || !w || !M || !R || !y || !Y32) return;
    if (Tin <= 0 || Cin <= 0 || Cout <= 0 || K <= 0 || stride <= 0 || dil <= 0) return;

    const int Tout = conv1d_out_len(Tin, K, stride, pad, dil);
    if (Tout <= 0) return;
    
    // Stage A+B: conv + requant to int8 (pre-activation)
    conv1d(
        x, Tin, Cin,
        w, b,
        Cout, K, stride, pad, dil,
        M, R,
        y, Tout, Y32
    );
    
    // Stage C: SiLU via LUT (in-place over y)
    // We have Cout*Tout contiguous elements in channel-major layout.
    silu_lut_q15_model(
        y, (size_t)(Cout * Tout),
        alpha, beta, rshift,
        y
    );
    
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CONVSTEM_H_
