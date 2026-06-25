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
#include "Simd/SimdSynetQuantizedInnerProduct.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCopy.h"
#include "Simd/SimdTile.h"

namespace Simd
{
#if defined(SIMD_AMXBF16_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace AmxBf16
    {
        typedef Simd::QuantizedInnerProductParam QipParam;
        typedef Base::SynetQuantizedInnerProductGemmV1::AlgParam AlgParam;
        typedef Base::SynetQuantizedInnerProductGemmV1::PrepPtr PrepPtr;
        typedef Base::SynetQuantizedInnerProductGemmV1::GemmPtr GemmPtr;

        //-------------------------------------------------------------------------------------------------

        static void QuantizedInnerProductGemmV1_PrepA_8u(const uint8_t* src, float norm, uint8_t zero, const QipParam& p, const AlgParam& a, size_t M, size_t, uint8_t* dst)
        {
            size_t KA = Simd::AlignLo(p.K, A);
            __mmask64 srcTail = TailMask64(p.K - KA), dstTail = TailMask64(a.aK - KA);
            for (size_t i = 0; i < M; ++i)
            {
                size_t k = 0;
                for (; k < KA; k += A)
                    Copy(src + k, dst + k);
                if (dstTail)
                    Copy(src + k, dst + k, srcTail, dstTail);
                src += p.K;
                dst += a.aK;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetQuantizedInnerProductGemmV1::SynetQuantizedInnerProductGemmV1(const QuantizedInnerProductParam& p)
            : Base::SynetQuantizedInnerProductGemmV1(p)
        {
            SetAlgParam();
            _prepA = QuantizedInnerProductGemmV1_PrepA_8u;
            //if (p.typeC == SimdTensorData8u)
            //    _gemm = QuantizedInnerProductGemmV0_2<Term8iLast8u>;
            //else
            //    _gemm = NULL;// QuantizedInnerProductGemmV0_2<Term8iLast32f>;
        }
    }
#endif
}
