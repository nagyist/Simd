/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
#include "Simd/SimdSynetQuantizedConvolution.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetQuantizedActivation.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"
#include "Simd/SimdTile.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdCopy.h"

namespace Simd
{
#if defined(SIMD_AMXBF16_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace AmxBf16
    {
        typedef Base::SynetQuantizedConvolutionNhwcGemmV1::AlgParam AlgParam;
        typedef Base::SynetQuantizedConvolutionNhwcGemmV1::GemmPtr GemmPtr;

        //-----------------------------------------------------------------------------------------

        static void QuantizedConvolutionNhwcGemmV1_ReorderD(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint8_t* dst)
        {
            size_t C = p.srcC, C64 = AlignLo(C, 64), K = a.bufK, kcX = p.kernelX * C;
            __mmask64 gM = TailMask64(K - a.K), cM = TailMask64(C - C64);
            __m512i _zero = _mm512_set1_epi8(zero);
            for (size_t dy = yBeg; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx, dst += K)
                {
                    uint8_t* pd = dst;
                    for (size_t ky = 0, k = 0; ky < p.kernelY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                        if (sy < p.srcH)
                        {
                            for (size_t kx = 0; kx < p.kernelX; kx++)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                if (sx < p.srcW)
                                    Copy(src + (sy * p.srcW + sx) * C, C64, cM, pd);
                                else
                                    SetZeros(pd, _zero, C64, cM);
                                pd += C;
                            }
                        }
                        else
                        {
                            SetZeros(pd, _zero, kcX);
                            pd += kcX;
                        }
                    }
                    SetZero(pd, _mm512_setzero_si512(), gM);
                }
            }
        }

        static void QuantizedConvolutionNhwcGemmV1_ReorderD1d(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint8_t* dst)
        {
            assert(p.IsDilation(1));
            size_t C = p.srcC, C64 = AlignLo(C, 64), K = a.bufK, kC = p.kernelX * C, kC64 = AlignLo(kC, 64), sX = p.strideX, cW = p.srcW * C, kY = p.kernelY, scX = sX * C;
            size_t dyB = DivHi(p.padY, p.strideY), dyE = p.dstH - DivHi(p.padH, p.strideY), dxB = DivHi(p.padX, p.strideX), dxE = p.dstW - DivHi(p.padW, p.strideX);
            __mmask64 gM = TailMask64(K - a.K), cM = TailMask64(C - C64), kcM = TailMask64(kC - kC64);
            __m512i _zero = _mm512_set1_epi8(zero);
            for (size_t dy = yBeg; dy < yEnd; ++dy)
            {
                size_t dx = 0;
                for (; dx < dxB; ++dx, dst += K)
                {
                    uint8_t* pd = dst;
                    ptrdiff_t sxcB = (dx * sX - p.padX) * C, sxcE = sxcB + kC;
                    for (size_t ky = 0, k = 0; ky < kY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky - p.padY;
                        if (sy < p.srcH)
                        {
                            for (ptrdiff_t sxc = sxcB; sxc < sxcE; sxc += C, pd += C)
                            {
                                if ((size_t)sxc < cW)
                                    Copy(src + sy * cW + sxc, C64, cM, pd);
                                else
                                    SetZeros(pd, _zero, C64, cM);
                            }
                        }
                        else
                        {
                            SetZeros(pd, _zero, kC64, kcM);
                            pd += kC;
                        }
                    }
                    SetZero(pd, _mm512_setzero_si512(), gM);
                }
                if (dy >= dyB && dy < dyE)
                {
                    const uint8_t* ps = src + (dy * p.strideY - p.padY) * cW + (dx * sX - p.padX) * C;
                    for (; dx < dxE; ++dx, dst += K, ps += scX)
                    {
                        uint8_t* pd = dst;
                        for (size_t ky = 0; ky < kY; ky++, pd += kC)
                            Copy(ps + ky * cW, kC64, kcM, pd);
                        SetZero(pd, _mm512_setzero_si512(), gM);
                    }
                }
                else
                {
                    for (; dx < dxE; ++dx, dst += K)
                    {
                        uint8_t* pd = dst;
                        ptrdiff_t sxcB = (dx * sX - p.padX) * C;
                        for (size_t ky = 0; ky < kY; ky++)
                        {
                            size_t sy = dy * p.strideY + ky - p.padY;
                            if (sy < p.srcH)
                                Copy(src + sy * cW + sxcB, kC64, kcM, pd);
                            else
                                SetZeros(pd, _zero, kC64, kcM);
                            pd += kC;
                        }
                        SetZero(pd, _mm512_setzero_si512(), gM);
                    }
                }
                for (; dx < p.dstW; ++dx, dst += K)
                {
                    uint8_t* pd = dst;
                    ptrdiff_t sxcB = (dx * sX - p.padX) * C, sxcE = sxcB + kC;
                    for (size_t ky = 0, k = 0; ky < kY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky - p.padY;
                        if (sy < p.srcH)
                        {
                            for (ptrdiff_t sxc = sxcB; sxc < sxcE; sxc += C, pd += C)
                            {
                                if ((size_t)sxc < cW)
                                    Copy(src + sy * cW + sxc, C64, cM, pd);
                                else
                                    SetZeros(pd, _zero, C64, cM);
                            }
                        }
                        else
                        {
                            SetZeros(pd, _zero, kC64, kcM);
                            pd += kC;
                        }
                    }
                    SetZero(pd, _mm512_setzero_si512(), gM);
                }
            }
        }

        static void QuantizedConvolutionNhwcGemmV1_ReorderD1d16c(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint8_t* dst)
        {
            assert(p.IsDilation(1) && p.srcC <= 16 && p.srcC * p.kernelX <= 64);
            size_t K = a.bufK, C = p.srcC, kcX = p.kernelX * C, sX = p.strideX, cW = p.srcW * C, cwH = cW * p.srcH, kY = p.kernelY, scX = sX * C;
            size_t dyB = DivHi(p.padY, p.strideY), dyE = p.dstH - DivHi(p.padH, p.strideY), dxB = DivHi(p.padX, p.strideX), dxE = p.dstW - DivHi(p.padW, p.strideX);
            __mmask64 gM = TailMask64(K - a.K), kcM = TailMask64(kcX);
            __mmask16 cM = TailMask16(C);
            __m512i _zero = _mm512_set1_epi8(zero);
            for (size_t dy = yBeg; dy < yEnd; ++dy)
            {
                size_t dx = 0;
                for (; dx < dxB; ++dx, dst += K)
                {
                    uint8_t* pd = dst;
                    ptrdiff_t sxcB = (dx * sX - p.padX) * C, sxcE = sxcB + kcX;
                    for (size_t ky = 0; ky < kY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky - p.padY;
                        if (sy < p.srcH)
                        {
                            for (ptrdiff_t sxc = sxcB; sxc < sxcE; sxc += C, pd += C)
                            {
                                if ((size_t)sxc < cW)
                                    _mm_mask_storeu_epi8(pd, cM, _mm_maskz_loadu_epi8(cM, src + sy * cW + sxc));
                                else
                                    _mm_mask_storeu_epi8(pd, cM, _mm512_castsi512_si128(_zero));
                            }
                        }
                        else
                        {
                            _mm512_mask_storeu_epi8(pd, kcM, _zero);
                            pd += kcX;
                        }
                    }
                    _mm512_mask_storeu_epi8(pd, gM, _mm512_setzero_si512());
                }
                if (dy >= dyB && dy < dyE)
                {
                    const uint8_t* ps = src + (dy * p.strideY - p.padY) * cW + (dx * sX - p.padX) * C;
                    for (; dx < dxE; ++dx, dst += K, ps += scX)
                    {
                        uint8_t* pd = dst;
                        for (size_t ky = 0; ky < kY; ky++, pd += kcX)
                            _mm512_mask_storeu_epi8(pd, kcM, _mm512_maskz_loadu_epi8(kcM, ps + ky * cW));
                        _mm512_mask_storeu_epi8(pd, gM, _mm512_setzero_si512());
                    }
                }
                else
                {
                    for (; dx < dxE; ++dx, dst += K)
                    {
                        uint8_t* pd = dst;
                        ptrdiff_t sxcB = (dx * sX - p.padX) * C;
                        for (size_t ky = 0; ky < kY; ky++)
                        {
                            size_t sy = dy * p.strideY + ky - p.padY;
                            if (sy < p.srcH)
                                _mm512_mask_storeu_epi8(pd, kcM, _mm512_maskz_loadu_epi8(kcM, src + sy * cW + sxcB));
                            else
                                _mm512_mask_storeu_epi8(pd, kcM, _zero);
                            pd += kcX;
                        }
                        _mm512_mask_storeu_epi8(pd, gM, _mm512_setzero_si512());
                    }
                }
                for (; dx < p.dstW; ++dx, dst += K)
                {
                    uint8_t* pd = dst;
                    ptrdiff_t sxcB = (dx * sX - p.padX) * C, sxcE = sxcB + kcX;
                    for (size_t ky = 0; ky < kY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky - p.padY;
                        if (sy < p.srcH)
                        {
                            for (ptrdiff_t sxc = sxcB; sxc < sxcE; sxc += C, pd += C)
                            {
                                if ((size_t)sxc < cW)
                                    _mm_mask_storeu_epi8(pd, cM, _mm_maskz_loadu_epi8(cM, src + sy * cW + sxc));
                                else
                                    _mm_mask_storeu_epi8(pd, cM, _mm512_castsi512_si128(_zero));
                            }
                        }
                        else
                        {
                            _mm512_mask_storeu_epi8(pd, kcM, _zero);
                            pd += kcX;
                        }
                    }
                    _mm512_mask_storeu_epi8(pd, gM, _mm512_setzero_si512());
                }
            }
        }

        static void QuantizedConvolutionNhwcGemmV1_ReorderR(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint8_t* dst)
        {
            assert(Aligned(p.srcC, 64));
            size_t K = a.bufK, C = p.srcC, kcX = p.kernelX * C;
            __m512i _zero = _mm512_set1_epi8(zero);
            for (size_t dy = yBeg, dr = 0; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx, ++dr)
                {
                    size_t drB = dr & (~15), drO = dr & 15;
                    uint8_t* row = dst + drB * K + drO * 64;
                    for (size_t ky = 0; ky < p.kernelY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                        if (sy < p.srcH)
                        {
                            for (size_t kx = 0; kx < p.kernelX; kx++)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                if (sx < p.srcW)
                                {
                                    const uint8_t* ps = src + (sy * p.srcW + sx) * p.srcC;
                                    for (size_t sc = 0; sc < C; sc += 64, row += 1024)
                                        Avx512bw::Copy(ps + sc, row);
                                }
                                else
                                {
                                    for (size_t sc = 0; sc < C; sc += 64, row += 1024)
                                        SetZero(row, _zero);
                                }
                            }
                        }
                        else
                        {
                            for (size_t sc = 0; sc < kcX; sc += 64, row += 1024)
                                SetZero(row, _zero);
                        }
                    }
                }
            }
        }

        static void QuantizedConvolutionNhwcGemmV1_Reorder1x1D(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t M, uint8_t* dst)
        {
            size_t srcC64 = AlignLo(p.srcC, 64);
            __mmask64 srcMask = TailMask64(p.srcC - srcC64);
            for (size_t i = 0; i < M; ++i)
            {
                size_t sc = 0;
                for (; sc < srcC64; sc += 64)
                    Avx512bw::Copy(src + sc, dst + sc);
                if(srcMask)
                    Avx512bw::Copy(src + sc, dst + sc, srcMask);
                src += p.srcC;
                dst += a.bufK;
            }
        }

        static void QuantizedConvolutionNhwcGemmV1_Reorder1x1R(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t M, uint8_t* dst)
        {
            size_t srcC64 = AlignLo(p.srcC, 64);
            __mmask64 srcMask = TailMask64(p.srcC - srcC64);
            __m512i _zero = _mm512_set1_epi8(zero);
            for (size_t i = 0; i < M; i += 16)
            {
                size_t m = Min(i + 16, M) - i;
                size_t sc = 0;
                for (; sc < srcC64; sc += 64)
                {
                    size_t j = 0;
                    for (; j < m; ++j)
                        Avx512bw::Copy(src + sc + j * p.srcC, dst + j * 64 + sc * 16);
                    for (; j < 16; ++j)
                        SetZero(dst + j * 64 + sc * 16, _mm512_setzero_si512());
                }
                if (srcC64 < p.srcC)
                {
                    size_t j = 0;
                    for (; j < m; ++j)
                        Avx512bw::Copy(src + sc + j * p.srcC, dst + j * 64 + sc * 16, srcMask);
                    for (; j < 16; ++j)
                        SetZero(dst + j * 64 + sc * 16, _mm512_setzero_si512());
                }
                src += p.srcC * 16;
                dst += a.bufK * 16;
            }
        }

        //-----------------------------------------------------------------------------------------


        typedef void (*QuantizedConvolutionNhwcGemmV1_GemmPtr)(const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstS, 
            size_t dstC, int update, const int8_t* weight0, const __m512i* sBias, const __m512* sNorm, const __m512i& iLo, const __m512i& iHi, 
            const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, int32_t* sum0, int32_t* buf0, uint8_t* dst);

        //-----------------------------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type, int flush, int apply> void QuantizedConvolutionNhwcGemmV1_Gemm(
            const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t N, size_t M, size_t K, int update, const int8_t* weight, const int32_t* sBias, 
            const float* sNorm, int32_t iZero, float iScale, const float* params, float dNorm, int32_t dZero, int32_t* sum, int32_t* buf, uint8_t* dst)
        {
            size_t n = 32, Mn = AlignLoAny(M, n), m = M - Mn;
            size_t dW = a.bufK * DF, dS = a.bufK, dB = (term == Term8iInterim ? a.dB : DF), dD = p.dstC * a.elem;

            __m512 _sNorm[2], _iScale, _params[2], _dNorm;
            __m512i _sBias[2], _dZero = _mm512_set1_epi32(dZero), _iLo, _iHi;
            if (type != SimdConvolutionActivationIdentity)
            {
                _iLo = _mm512_set1_epi32(-iZero);
                _iHi = _mm512_set1_epi32(255 - iZero);
                _iScale = _mm512_set1_ps(iScale);
                _dNorm = _mm512_set1_ps(dNorm);
                _params[0] = _mm512_set1_ps(params[0]);
                _params[1] = _mm512_set1_ps(params[1]);
            }
            if (Mn)
            {
                SetTileConfFull();
                for (size_t j = 0; j < N; j += DF)
                {
                    size_t dN = Simd::Min(DF, N - j);
                    _sBias[0] = _mm512_loadu_si512((__m512i*)(sBias + j) + 0);
                    _sBias[1] = _mm512_loadu_si512((__m512i*)(sBias + j) + 1);
                    _sNorm[0] = _mm512_loadu_ps(sNorm + j + 0);
                    _sNorm[1] = _mm512_loadu_ps(sNorm + j + F);
                    if (type == SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + j + 0);
                        _params[1] = _mm512_loadu_ps(params + j + F);
                    }
                    __mmask32 tailD = term == Term8iLast8u ? TailMask32(dN) : (__mmask32)TailMask16(dN - AlignLo(dN - 1, 16));
                    buf = (term == Term8iInterim ? sum + j : buf);
            //        if (dN > F)
            //            Convolution16bNhwcGemmV2_GemmNxMx2<term, type, flush, 2, apply>(src, p, a, K, M, zero, weight, _bias, _params, sum + j, buf, dst + j * a.elem, tailD);
            //        else
            //            Convolution16bNhwcGemmV2_GemmNxMx2<term, type, flush, 1, apply>(src, p, a, K, M, zero, weight, _bias, _params, sum + j, buf, dst + j * a.elem, tailD);
            //        weight += dW;
                }
            }
            else
            {
            //    Convolution16bNhwcGemmV2_GemmPtr tail_2 = m > 16 ? Convolution16bNhwcGemmV2_Gemm2x2<term, type, flush, 0> : Convolution16bNhwcGemmV2_Gemm1x2<term, type, flush, 0>;
            //    Convolution16bNhwcGemmV2_GemmPtr tail_1 = m > 16 ? Convolution16bNhwcGemmV2_Gemm2x1<term, type, flush, 0> : Convolution16bNhwcGemmV2_Gemm1x1<term, type, flush, 0>;
                if (m > 16)
                    SetTileConf2x2(m, 32);
                else
                    SetTileConf1x2(m, 32);
                for (size_t j = 0; j < N; j += DF)
                {
                    size_t dN = Simd::Min(DF, N - j);
                    _sBias[0] = _mm512_loadu_si512((__m512i*)(sBias + j) + 0);
                    _sBias[1] = _mm512_loadu_si512((__m512i*)(sBias + j) + 1);
                    _sNorm[0] = _mm512_loadu_ps(sNorm + j + 0);
                    _sNorm[1] = _mm512_loadu_ps(sNorm + j + F);
                    if (type == SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + j + 0);
                        _params[1] = _mm512_loadu_ps(params + j + F);
                    }
                    __mmask32 tailD = term == Term8iLast8u ? TailMask32(dN) : (__mmask32)TailMask16(dN - AlignLo(dN - 1, 16));
                    buf = (term == Term8iInterim ? sum + j : buf);
            //        if (dN > F)
            //            tail_2(src, p, a, K, m, dN, zero, weight, _bias, _params, sum + j, buf, dB, dst + j * a.elem, tailD);
            //        else
            //            tail_1(src, p, a, K, m, dN, zero, weight, _bias, _params, sum + j, buf, dB, dst + j * a.elem, tailD);
            //        weight += dW;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        template <SimdConvolutionActivationType type, int apply> SIMD_INLINE void SetGemm(const ConvParam& p, const AlgParam& a, GemmPtr* gemm)
        {
            gemm[0] = QuantizedConvolutionNhwcGemmV1_Gemm<Term8iInterim, SimdConvolutionActivationIdentity, 0, 0>;
            if (p.dstT == SimdTensorData8u)
                gemm[1] = QuantizedConvolutionNhwcGemmV1_Gemm<Term8iLast8u, type, 0, apply>;
            else
                gemm[1] = QuantizedConvolutionNhwcGemmV1_Gemm<Term8iLast32f, type, 0, apply>;
        }

        template <SimdConvolutionActivationType type> SIMD_INLINE void SetGemm(const ConvParam& p, const AlgParam& a, GemmPtr* gemm)
        {
            size_t lastMacroK = a.bufK - AlignLoAny(a.bufK - 1, a.macroK);
            if (lastMacroK > 448)
                SetGemm<type, 1>(p, a, gemm);
            else if (lastMacroK > 192)
                SetGemm<type, 2>(p, a, gemm);
            else if (lastMacroK > 64)
                SetGemm<type, 4>(p, a, gemm);
            else
                SetGemm<type, 8>(p, a, gemm);
        }

        SynetQuantizedConvolutionNhwcGemmV1::SynetQuantizedConvolutionNhwcGemmV1(const ConvParam& p)
            : Base::SynetQuantizedConvolutionNhwcGemmV1(p)
        {
            SetAlgParam();
            AlgParam& a = _alg;
            if (_is1x1)
            {
                _convAny = NULL;
                if(a.K == a.bufK)
                    _conv1x1 = NULL;
                else if (a.batch == 1)
                {
                    _conv1x1 = QuantizedConvolutionNhwcGemmV1_Reorder1x1R;
                    a.reorderType = 1;
                }
                else
                {
                    _conv1x1 = QuantizedConvolutionNhwcGemmV1_Reorder1x1D;
                    a.reorderType = 0;
                }
            }
            else
            {
                _conv1x1 = NULL;
                if (Aligned(p.srcC, 64) && a.batch == 1 && Aligned(p.dstW, a.F))
                {
                    _convAny = QuantizedConvolutionNhwcGemmV1_ReorderR;
                    a.reorderType = 1;
                }
                else if (p.IsDilation(1) && p.srcC <= 16 && p.srcC * p.kernelX <= 64)
                    _convAny = QuantizedConvolutionNhwcGemmV1_ReorderD1d16c;
                else if (p.IsDilation(1))
                    _convAny = QuantizedConvolutionNhwcGemmV1_ReorderD1d;
                else
                    _convAny = QuantizedConvolutionNhwcGemmV1_ReorderD;
            }
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetGemm<SimdConvolutionActivationIdentity>(p, _alg, _gemm); break;
            case SimdConvolutionActivationRelu: SetGemm<SimdConvolutionActivationRelu>(p, _alg, _gemm); break;
            case SimdConvolutionActivationLeakyRelu: SetGemm<SimdConvolutionActivationLeakyRelu>(p, _alg, _gemm); break;
            case SimdConvolutionActivationRestrictRange: SetGemm<SimdConvolutionActivationRestrictRange>(p, _alg, _gemm); break;
            case SimdConvolutionActivationPrelu: SetGemm<SimdConvolutionActivationPrelu>(p, _alg, _gemm); break;
            case SimdConvolutionActivationElu: SetGemm<SimdConvolutionActivationElu>(p, _alg, _gemm); break;
            case SimdConvolutionActivationHswish: SetGemm<SimdConvolutionActivationHswish>(p, _alg, _gemm); break;
            case SimdConvolutionActivationMish: SetGemm<SimdConvolutionActivationMish>(p, _alg, _gemm); break;
            case SimdConvolutionActivationHardSigmoid: SetGemm<SimdConvolutionActivationHardSigmoid>(p, _alg, _gemm); break;
            case SimdConvolutionActivationSwish: SetGemm<SimdConvolutionActivationSwish>(p, _alg, _gemm); break;
            case SimdConvolutionActivationGelu: SetGemm<SimdConvolutionActivationGelu>(p, _alg, _gemm); break;
            default: assert(0);
            }
        }
    }
#endif
}
