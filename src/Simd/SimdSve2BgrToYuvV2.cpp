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
#include "Simd/SimdMemory.h"
#include "Simd/SimdYuvToBgr.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svint16_t PackI32ToI16(const svint32_t& lo, const svint32_t& hi)
        {
            return svqxtnt_s32(svqxtnb_s32(lo), hi);
        }

        SIMD_INLINE svuint8_t PackSequentialI16ToU8(const svint16_t& value)
        {
            return PackSaturatedI16ToU8(svuzp1_s16(value, value), svuzp2_s16(value, value));
        }

        //-------------------------------------------------------------------------------------------------

        template <class T> SIMD_INLINE void BgrToYuv444pV2(const uint8_t* bgr, uint8_t* y, uint8_t* u, uint8_t* v, const svbool_t& mask)
        {
            svuint8x3_t _bgr = svld3_u8(mask, bgr);
            svuint8_t blue = svget3(_bgr, 0);
            svuint8_t green = svget3(_bgr, 1);
            svuint8_t red = svget3(_bgr, 2);
            svst1_u8(mask, y, BgrToY8<T>(blue, green, red));
            svst1_u8(mask, u, BgrToU8<T>(blue, green, red));
            svst1_u8(mask, v, BgrToV8<T>(blue, green, red));
        }

        template <class T> void BgrToYuv444pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            size_t A = svlen(svuint8_t()), A3 = A * 3;
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            for (size_t row = 0; row < height; ++row)
            {
                size_t colBgr = 0, col = 0;
                for (; col < widthA; colBgr += A3, col += A)
                    BgrToYuv444pV2<T>(bgr + colBgr, y + col, u + col, v + col, body);
                if (col < width)
                    BgrToYuv444pV2<T>(bgr + colBgr, y + col, u + col, v + col, tail);
                bgr += bgrStride;
                y += yStride;
                u += uStride;
                v += vStride;
            }
        }

        void BgrToYuv444pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgrToYuv444pV2<Base::Bt601>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgrToYuv444pV2<Base::Bt709>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgrToYuv444pV2<Base::Bt2020>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgrToYuv444pV2<Base::Trect871>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <class T> SIMD_INLINE void BgrToYuv420pV2(const uint8_t* bgr0, size_t bgrStride, uint8_t* y0, size_t yStride,
            uint8_t* u, uint8_t* v, const svbool_t& maskY, const svbool_t& maskUv)
        {
            const uint8_t* bgr1 = bgr0 + bgrStride;
            uint8_t* y1 = y0 + yStride;

            svuint8x3_t bgr00 = svld3_u8(maskY, bgr0);
            svuint8x3_t bgr10 = svld3_u8(maskY, bgr1);

            svst1_u8(maskY, y0, BgrToY8<T>(svget3(bgr00, 0), svget3(bgr00, 1), svget3(bgr00, 2)));
            svst1_u8(maskY, y1, BgrToY8<T>(svget3(bgr10, 0), svget3(bgr10, 1), svget3(bgr10, 2)));

            svuint16_t blue = AverageUv(svget3(bgr00, 0), svget3(bgr10, 0), maskY);
            svuint16_t green = AverageUv(svget3(bgr00, 1), svget3(bgr10, 1), maskY);
            svuint16_t red = AverageUv(svget3(bgr00, 2), svget3(bgr10, 2), maskY);

            svst1_u8(maskUv, u, PackSequentialI16ToU8(BgrToU16<T>(svreinterpret_s16_u16(blue), svreinterpret_s16_u16(green), svreinterpret_s16_u16(red))));
            svst1_u8(maskUv, v, PackSequentialI16ToU8(BgrToV16<T>(svreinterpret_s16_u16(blue), svreinterpret_s16_u16(green), svreinterpret_s16_u16(red))));
        }

        template <class T> void BgrToYuv420pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            size_t A = svlen(svuint8_t()), HA = A / 2, A3 = A * 3;
            size_t widthA = AlignLo(width, A), tail = width - widthA;
            const svbool_t bodyY = svptrue_b8();
            const svbool_t bodyUv = svwhilelt_b8(size_t(0), HA);
            const svbool_t tailY = svwhilelt_b8(size_t(0), tail);
            const svbool_t tailUv = svwhilelt_b8(size_t(0), tail / 2);
            for (size_t row = 0; row < height; row += 2)
            {
                size_t colBgr = 0, colY = 0, colUv = 0;
                for (; colY < widthA; colBgr += A3, colY += A, colUv += HA)
                    BgrToYuv420pV2<T>(bgr + colBgr, bgrStride, y + colY, yStride, u + colUv, v + colUv, bodyY, bodyUv);
                if (colY < width)
                    BgrToYuv420pV2<T>(bgr + colBgr, bgrStride, y + colY, yStride, u + colUv, v + colUv, tailY, tailUv);
                bgr += 2 * bgrStride;
                y += 2 * yStride;
                u += uStride;
                v += vStride;
            }
        }

        void BgrToYuv420pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgrToYuv420pV2<Base::Bt601>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgrToYuv420pV2<Base::Bt709>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgrToYuv420pV2<Base::Bt2020>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgrToYuv420pV2<Base::Trect871>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <class T> SIMD_INLINE void BgrToYuv422pV2(const uint8_t* bgr, uint8_t* y, uint8_t* u, uint8_t* v,
            const svbool_t& maskY, const svbool_t& maskUv)
        {
            svuint8x3_t _bgr = svld3_u8(maskY, bgr);

            svst1_u8(maskY, y, BgrToY8<T>(svget3(_bgr, 0), svget3(_bgr, 1), svget3(_bgr, 2)));

            svuint16_t blue = AverageUv(svget3(_bgr, 0), maskY);
            svuint16_t green = AverageUv(svget3(_bgr, 1), maskY);
            svuint16_t red = AverageUv(svget3(_bgr, 2), maskY);

            svst1_u8(maskUv, u, PackSequentialI16ToU8(BgrToU16<T>(svreinterpret_s16_u16(blue), svreinterpret_s16_u16(green), svreinterpret_s16_u16(red))));
            svst1_u8(maskUv, v, PackSequentialI16ToU8(BgrToV16<T>(svreinterpret_s16_u16(blue), svreinterpret_s16_u16(green), svreinterpret_s16_u16(red))));
        }

        template <class T> void BgrToYuv422pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (width >= 2));

            size_t A = svlen(svuint8_t()), HA = A / 2, A3 = A * 3;
            size_t widthA = AlignLo(width, A), tail = width - widthA;
            const svbool_t bodyY = svptrue_b8();
            const svbool_t bodyUv = svwhilelt_b8(size_t(0), HA);
            const svbool_t tailY = svwhilelt_b8(size_t(0), tail);
            const svbool_t tailUv = svwhilelt_b8(size_t(0), tail / 2);
            for (size_t row = 0; row < height; ++row)
            {
                size_t colBgr = 0, colY = 0, colUv = 0;
                for (; colY < widthA; colBgr += A3, colY += A, colUv += HA)
                    BgrToYuv422pV2<T>(bgr + colBgr, y + colY, u + colUv, v + colUv, bodyY, bodyUv);
                if (colY < width)
                    BgrToYuv422pV2<T>(bgr + colBgr, y + colY, u + colUv, v + colUv, tailY, tailUv);
                bgr += bgrStride;
                y += yStride;
                u += uStride;
                v += vStride;
            }
        }

        void BgrToYuv422pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgrToYuv422pV2<Base::Bt601>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgrToYuv422pV2<Base::Bt709>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgrToYuv422pV2<Base::Bt2020>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgrToYuv422pV2<Base::Trect871>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }
    }
#endif
}
