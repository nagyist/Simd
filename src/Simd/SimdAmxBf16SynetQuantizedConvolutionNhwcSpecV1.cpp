/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdTile.h"
#include "Simd/SimdCopy.h"

namespace Simd
{
#if defined(SIMD_AMXBF16_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace AmxBf16
    {
        typedef Base::SynetQuantizedConvolutionNhwcSpecV1::AlgParam AlgParam;
        typedef Base::SynetQuantizedConvolutionNhwcSpecV1::LastConvPtr LastConvPtr;

        //-----------------------------------------------------------------------------------------

        static void QuantizedConvolutionNhwcSpecV1_Reorder(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, int end, uint8_t* dst)
        {
            assert(a.microC == A);
            size_t sC = p.srcC, sCA = Simd::AlignLo(sC, A), asC = a.srcC, sW = p.srcW, asW = a.srcW;
            size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? p.srcH : dyEnd + syPad), apH = a.padH;
            size_t cD = a.batch * a.srcH * a.srcW + a.padE, sD = a.microC;
            __m512i _zero = _mm512_set1_epi8(zero);
            __mmask64 tailC = TailMask64(sC - sCA);
            if (dyBeg == 0)
            {
                for (size_t s = 0, n = a.padV * asW; s < n; ++s)
                    for (size_t c = 0; c < asC; c += A)
                        _mm512_storeu_si512((__m512i*)(dst + c * cD + s * sD), _zero);
                dst += a.padV * asW * sD;
                syBeg = 0;
            }
            else
            {
                syBeg = dyBeg + syPad;
                src += syBeg * sW * sC;
                dst += (dyBeg + p.kernelY - 1 + a.padV - p.padY) * asW * sD;
            }
            for (size_t sy = syBeg; sy < syEnd; ++sy)
            {
                if (apH)
                {
                    for (size_t s = 0; s < apH; ++s)
                        for (size_t c = 0; c < asC; c += A)
                            _mm512_storeu_si512((__m512i*)(dst + c * cD + s * sD), _zero);
                    dst += apH * sD;
                }
                for (size_t sx = 0; sx < sW; ++sx)
                {
                    size_t sc = 0;
                    for (; sc < sCA; sc += A)
                        Copy(src + sc, dst + sc * cD);
                    if (tailC)
                        Copy(src + sc, dst + sc * cD, tailC);
                    src += sC;
                    dst += sD;
                }
            }
            if (end)
            {
                for (size_t s = 0, n = a.padE; s < n; ++s)
                    for (size_t c = 0; c < asC; c += A)
                        _mm512_storeu_si512((__m512i*)(dst + c * cD + s * sD), _zero);
            }
            else if (dyEnd != p.dstH)
            {
                for (size_t s = 0; s < apH; ++s)
                    for (size_t c = 0; c < asC; c += A)
                        _mm512_storeu_si512((__m512i*)(dst + c * cD + s * sD), _zero);
            }
        }

        static void QuantizedConvolutionNhwcSpecV1_Reorder64(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, int end, uint8_t* dst)
        {
            assert(a.microC == A && p.srcC <= A);
            size_t sC = p.srcC, sW = p.srcW, asW = a.srcW;
            size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? p.srcH : dyEnd + syPad), apH = a.padH;
            size_t cD = a.batch * a.srcH * a.srcW + a.padE, sD = a.microC;
            __m512i _zero = _mm512_set1_epi8(zero);
            __mmask64 tailC = TailMask64(sC);
            if (dyBeg == 0)
            {
                for (size_t s = 0, n = a.padV * asW; s < n; ++s, dst += A)
                    _mm512_storeu_si512((__m512i*)dst, _zero);
                syBeg = 0;
            }
            else
            {
                syBeg = dyBeg + syPad;
                src += syBeg * sW * sC;
                dst += (dyBeg + p.kernelY - 1 + a.padV - p.padY) * asW * sD;
            }
            for (size_t sy = syBeg; sy < syEnd; ++sy)
            {
                for (size_t s = 0; s < apH; ++s, dst += A)
                    _mm512_storeu_si512((__m512i*)dst, _zero);
                for (size_t sx = 0; sx < sW; ++sx, src += sC, dst += A)
                    Copy(src, dst, tailC);
            }
            if (end)
            {
                for (size_t s = 0, n = a.padE; s < n; ++s, dst += A)
                    _mm512_storeu_si512((__m512i*)dst, _zero);
            }
            else if (dyEnd != p.dstH)
            {
                for (size_t s = 0; s < apH; ++s, dst += A)
                    _mm512_storeu_si512((__m512i*)dst, _zero);
            }
        }


        //-----------------------------------------------------------------------------------------


        //-----------------------------------------------------------------------------------------

        SynetQuantizedConvolutionNhwcSpecV1::SynetQuantizedConvolutionNhwcSpecV1(const ConvParam& p)
            : Base::SynetQuantizedConvolutionNhwcSpecV1(p)
        {
            SetAlgParam();
            AlgParam& a = _alg;
            if (_src8u)
            {
                if (p.srcC <= 64)
                    _preprocess = QuantizedConvolutionNhwcSpecV1_Reorder64;
                else
                    _preprocess = QuantizedConvolutionNhwcSpecV1_Reorder;
            }
            else
                assert(0);
            //_convolution = QuantizedConvolutionNhwcSpecV0_2;
        }
    }
#endif
}
