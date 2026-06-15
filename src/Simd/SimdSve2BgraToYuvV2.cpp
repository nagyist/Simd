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

        SIMD_INLINE svuint8_t PackSaturatedI16ToU8(const svint16_t& lo, const svint16_t& hi)
        {
            return svqxtunt_s16(svqxtunb_s16(lo), hi);
        }

        SIMD_INLINE svuint8_t PackSequentialI16ToU8(const svint16_t& value)
        {
            return PackSaturatedI16ToU8(svuzp1_s16(value, value), svuzp2_s16(value, value));
        }

        template<class T> SIMD_INLINE svint16_t BgrToY16(const svint16_t& blue, const svint16_t& green, const svint16_t& red)
        {
            svint32_t yb = svmlalb_n_s32(svmlalb_n_s32(svmlalb_n_s32(svdup_n_s32(T::B_ROUND), blue, T::B_2_Y), green, T::G_2_Y), red, T::R_2_Y);
            svint32_t yt = svmlalt_n_s32(svmlalt_n_s32(svmlalt_n_s32(svdup_n_s32(T::B_ROUND), blue, T::B_2_Y), green, T::G_2_Y), red, T::R_2_Y);
            return svadd_n_s16_x(svptrue_b32(), svqrshrnt_n_s32(svqrshrnb_n_s32(yb, T::B_SHIFT), yt, T::B_SHIFT), T::Y_LO);
        }

        template<class T> SIMD_INLINE svuint8_t BgrToY8(const svuint8_t& blue, const svuint8_t& green, const svuint8_t& red)
        {
            return PackSaturatedI16ToU8(
                BgrToY16<T>(svreinterpret_s16_u16(svmovlb_u16(blue)), svreinterpret_s16_u16(svmovlb_u16(green)), svreinterpret_s16_u16(svmovlb_u16(red))),
                BgrToY16<T>(svreinterpret_s16_u16(svmovlt_u16(blue)), svreinterpret_s16_u16(svmovlt_u16(green)), svreinterpret_s16_u16(svmovlt_u16(red))));
        }

        template<class T> SIMD_INLINE svint32_t BgrToU32(const svuint32_t& blue, const svuint32_t& green, const svuint32_t& red)
        {
            const svbool_t mask = svptrue_b32();
            svint32_t u = svdup_n_s32(T::B_ROUND);
            u = svadd_s32_x(mask, u, svmul_n_s32_x(mask, svreinterpret_s32_u32(blue), T::B_2_U));
            u = svadd_s32_x(mask, u, svmul_n_s32_x(mask, svreinterpret_s32_u32(green), T::G_2_U));
            u = svadd_s32_x(mask, u, svmul_n_s32_x(mask, svreinterpret_s32_u32(red), T::R_2_U));
            return svasr_n_s32_x(mask, u, T::B_SHIFT);
        }

        template<class T> SIMD_INLINE svint16_t BgrToU16(const svuint16_t& blue, const svuint16_t& green, const svuint16_t& red)
        {
            const svbool_t mask = svptrue_b16();
            return svadd_n_s16_x(mask, PackI32ToI16(
                BgrToU32<T>(svmovlb_u32(blue), svmovlb_u32(green), svmovlb_u32(red)),
                BgrToU32<T>(svmovlt_u32(blue), svmovlt_u32(green), svmovlt_u32(red))), T::UV_Z);
        }

        template<class T> SIMD_INLINE svuint8_t BgrToU8(const svuint8_t& blue, const svuint8_t& green, const svuint8_t& red)
        {
            return PackSaturatedI16ToU8(
                BgrToU16<T>(svmovlb_u16(blue), svmovlb_u16(green), svmovlb_u16(red)),
                BgrToU16<T>(svmovlt_u16(blue), svmovlt_u16(green), svmovlt_u16(red)));
        }

        template<class T> SIMD_INLINE svint32_t BgrToV32(const svuint32_t& blue, const svuint32_t& green, const svuint32_t& red)
        {
            const svbool_t mask = svptrue_b32();
            svint32_t v = svdup_n_s32(T::B_ROUND);
            v = svadd_s32_x(mask, v, svmul_n_s32_x(mask, svreinterpret_s32_u32(blue), T::B_2_V));
            v = svadd_s32_x(mask, v, svmul_n_s32_x(mask, svreinterpret_s32_u32(green), T::G_2_V));
            v = svadd_s32_x(mask, v, svmul_n_s32_x(mask, svreinterpret_s32_u32(red), T::R_2_V));
            return svasr_n_s32_x(mask, v, T::B_SHIFT);
        }

        template<class T> SIMD_INLINE svint16_t BgrToV16(const svuint16_t& blue, const svuint16_t& green, const svuint16_t& red)
        {
            const svbool_t mask = svptrue_b16();
            return svadd_n_s16_x(mask, PackI32ToI16(
                BgrToV32<T>(svmovlb_u32(blue), svmovlb_u32(green), svmovlb_u32(red)),
                BgrToV32<T>(svmovlt_u32(blue), svmovlt_u32(green), svmovlt_u32(red))), T::UV_Z);
        }

        template<class T> SIMD_INLINE svuint8_t BgrToV8(const svuint8_t& blue, const svuint8_t& green, const svuint8_t& red)
        {
            return PackSaturatedI16ToU8(
                BgrToV16<T>(svmovlb_u16(blue), svmovlb_u16(green), svmovlb_u16(red)),
                BgrToV16<T>(svmovlt_u16(blue), svmovlt_u16(green), svmovlt_u16(red)));
        }

        //-------------------------------------------------------------------------------------------------

        template <class T> SIMD_INLINE void BgraToYuv444pV2(const uint8_t* bgra, uint8_t* y, uint8_t* u, uint8_t* v, const svbool_t& mask)
        {
            svuint8x4_t _bgra = svld4_u8(mask, bgra);
            svuint8_t blue = svget4(_bgra, 0);
            svuint8_t green = svget4(_bgra, 1);
            svuint8_t red = svget4(_bgra, 2);
            svst1_u8(mask, y, BgrToY8<T>(blue, green, red));
            svst1_u8(mask, u, BgrToU8<T>(blue, green, red));
            svst1_u8(mask, v, BgrToV8<T>(blue, green, red));
        }

        template <class T> void BgraToYuv444pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            size_t A = svlen(svuint8_t()), A4 = A * 4;
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            for (size_t row = 0; row < height; ++row)
            {
                size_t colBgra = 0, col = 0;
                for (; col < widthA; colBgra += A4, col += A)
                    BgraToYuv444pV2<T>(bgra + colBgra, y + col, u + col, v + col, body);
                if (col < width)
                    BgraToYuv444pV2<T>(bgra + colBgra, y + col, u + col, v + col, tail);
                bgra += bgraStride;
                y += yStride;
                u += uStride;
                v += vStride;
            }
        }

        void BgraToYuv444pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgraToYuv444pV2<Base::Bt601>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgraToYuv444pV2<Base::Bt709>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgraToYuv444pV2<Base::Bt2020>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgraToYuv444pV2<Base::Trect871>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }

        template <class T> SIMD_INLINE void BgraToYuv420pV2(const uint8_t* bgra0, size_t bgraStride, uint8_t* y0, size_t yStride,
            uint8_t* u, uint8_t* v, const svbool_t& maskY, const svbool_t& maskUv)
        {
            const uint8_t* bgra1 = bgra0 + bgraStride;
            uint8_t* y1 = y0 + yStride;

            svuint8x4_t bgra00 = svld4_u8(maskY, bgra0);
            svuint8x4_t bgra10 = svld4_u8(maskY, bgra1);

            svst1_u8(maskY, y0, BgrToY8<T>(svget4(bgra00, 0), svget4(bgra00, 1), svget4(bgra00, 2)));
            svst1_u8(maskY, y1, BgrToY8<T>(svget4(bgra10, 0), svget4(bgra10, 1), svget4(bgra10, 2)));

            svuint16_t blue = AverageUv(svget4(bgra00, 0), svget4(bgra10, 0), maskY);
            svuint16_t green = AverageUv(svget4(bgra00, 1), svget4(bgra10, 1), maskY);
            svuint16_t red = AverageUv(svget4(bgra00, 2), svget4(bgra10, 2), maskY);

            svst1_u8(maskUv, u, PackSequentialI16ToU8(BgrToU16<T>(blue, green, red)));
            svst1_u8(maskUv, v, PackSequentialI16ToU8(BgrToV16<T>(blue, green, red)));
        }

        template <class T> void BgraToYuv420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            size_t A = svlen(svuint8_t()), HA = A / 2, A4 = A * 4;
            size_t widthA = AlignLo(width, A), tail = width - widthA;
            const svbool_t bodyY = svptrue_b8();
            const svbool_t bodyUv = svwhilelt_b8(size_t(0), HA);
            const svbool_t tailY = svwhilelt_b8(size_t(0), tail);
            const svbool_t tailUv = svwhilelt_b8(size_t(0), tail / 2);
            for (size_t row = 0; row < height; row += 2)
            {
                size_t colBgra = 0, colY = 0, colUv = 0;
                for (; colY < widthA; colBgra += A4, colY += A, colUv += HA)
                    BgraToYuv420pV2<T>(bgra + colBgra, bgraStride, y + colY, yStride, u + colUv, v + colUv, bodyY, bodyUv);
                if (colY < width)
                    BgraToYuv420pV2<T>(bgra + colBgra, bgraStride, y + colY, yStride, u + colUv, v + colUv, tailY, tailUv);
                bgra += 2 * bgraStride;
                y += 2 * yStride;
                u += uStride;
                v += vStride;
            }
        }

        void BgraToYuv420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgraToYuv420pV2<Base::Bt601>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgraToYuv420pV2<Base::Bt709>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgraToYuv420pV2<Base::Bt2020>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgraToYuv420pV2<Base::Trect871>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <class T> SIMD_INLINE void BgraToYuv422pV2(const uint8_t* bgra, uint8_t* y, uint8_t* u, uint8_t* v,
            const svbool_t& maskY, const svbool_t& maskUv)
        {
            svuint8x4_t _bgra = svld4_u8(maskY, bgra);

            svst1_u8(maskY, y, BgrToY8<T>(svget4(_bgra, 0), svget4(_bgra, 1), svget4(_bgra, 2)));

            svuint16_t blue = AverageUv(svget4(_bgra, 0), maskY);
            svuint16_t green = AverageUv(svget4(_bgra, 1), maskY);
            svuint16_t red = AverageUv(svget4(_bgra, 2), maskY);

            svst1_u8(maskUv, u, PackSequentialI16ToU8(BgrToU16<T>(blue, green, red)));
            svst1_u8(maskUv, v, PackSequentialI16ToU8(BgrToV16<T>(blue, green, red)));
        }

        template <class T> void BgraToYuv422pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (width >= 2));

            size_t A = svlen(svuint8_t()), HA = A / 2, A4 = A * 4;
            size_t widthA = AlignLo(width, A), tail = width - widthA;
            const svbool_t bodyY = svptrue_b8();
            const svbool_t bodyUv = svwhilelt_b8(size_t(0), HA);
            const svbool_t tailY = svwhilelt_b8(size_t(0), tail);
            const svbool_t tailUv = svwhilelt_b8(size_t(0), tail / 2);
            for (size_t row = 0; row < height; ++row)
            {
                size_t colBgra = 0, colY = 0, colUv = 0;
                for (; colY < widthA; colBgra += A4, colY += A, colUv += HA)
                    BgraToYuv422pV2<T>(bgra + colBgra, y + colY, u + colUv, v + colUv, bodyY, bodyUv);
                if (colY < width)
                    BgraToYuv422pV2<T>(bgra + colBgra, y + colY, u + colUv, v + colUv, tailY, tailUv);
                bgra += bgraStride;
                y += yStride;
                u += uStride;
                v += vStride;
            }
        }

        void BgraToYuv422pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgraToYuv422pV2<Base::Bt601>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgraToYuv422pV2<Base::Bt709>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgraToYuv422pV2<Base::Bt2020>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgraToYuv422pV2<Base::Trect871>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }
    }
#endif
}
