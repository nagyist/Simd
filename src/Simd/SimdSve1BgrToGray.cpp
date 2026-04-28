/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar,
*               2022-2022 Fabien Spindler,
*               2022-2022 Souriya Trinh.
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
#include "Simd/SimdSve1.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdUnpack.h"
#include "Simd/SimdConversion.h"
#include "Simd/SimdNeon.h"

namespace Simd
{
#ifdef SIMD_SVE_ENABLE
    namespace Sve
    {
        //SIMD_INLINE svuint8_t BgrToGray(svint8x3_t bgr)
        //{
        //    svuint8x8_t lo = vmovn_u16(BgrToGray(UnpackU8<0>(bgr.val[0]), UnpackU8<0>(bgr.val[1]), UnpackU8<0>(bgr.val[2])));
        //    uint8x8_t hi = vmovn_u16(BgrToGray(UnpackU8<1>(bgr.val[0]), UnpackU8<1>(bgr.val[1]), UnpackU8<1>(bgr.val[2])));
        //    return vcombine_u8(lo, hi);
        //}

        //void BgrToGray(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* gray, size_t grayStride)
        //{
        //    size_t A = svlen(svuint8_t()), A3 = A * 3;
        //    size_t widthA = AlignLo(width, A);
        //    const svbool_t body = svwhilelt_b8(size_t(0), A);
        //    const svbool_t tail = svwhilelt_b8(widthA, width);
        //    for (size_t row = 0; row < height; ++row)
        //    {
        //        size_t col = 0, offset = 0;
        //        for (; col < widthA; col += A, offset += A3)
        //        {
        //            svuint8x3_t _bgr = svld3_u8(body, bgr + offset);
        //            svst1_u8(body, gray + col, BgrToGray(_bgr));
        //        }
        //        if (widthA < width)
        //        {
        //            svuint8x3_t _bgr = svld3_u8(tail, bgr + offset);
        //            svst1_u8(tail, gray + col, BgrToGray(_bgr));
        //        }
        //        bgr += bgrStride;
        //        gray += grayStride;
        //    }
        //}
    }
#endif
}
