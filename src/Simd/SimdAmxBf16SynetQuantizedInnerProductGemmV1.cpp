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
#include "Simd/SimdSet.h"

namespace Simd
{
#if defined(SIMD_AMXBF16_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace AmxBf16
    {
        typedef Simd::QuantizedInnerProductParam QipParam;
        typedef Base::SynetQuantizedInnerProductGemmV1::AlgParam AlgParam;
        typedef Base::SynetQuantizedInnerProductGemmV1::PrepPtr PrepPtr;

        //-------------------------------------------------------------------------------------------------

        static void QuantizedInnerProductGemmV1_PrepA_8uD(const uint8_t* src, float norm, uint8_t zero, const QipParam& p, const AlgParam& a, size_t M, size_t, uint8_t* dst)
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

        static void QuantizedInnerProductGemmV1_PrepA_8uR(const uint8_t* src, float norm, uint8_t zero, const QipParam& p, const AlgParam& a, size_t M, size_t, uint8_t* dst)
        {
            size_t K = p.K, K64 = AlignLo(K, 64);
            __mmask64 srcMask = TailMask64(K - K64);
            __m512i _zero = _mm512_set1_epi8(zero);
            for (size_t i = 0; i < M; i += 16)
            {
                size_t m = Min(i + 16, M) - i;
                size_t k = 0;
                for (; k < K64; k += 64)
                {
                    size_t j = 0;
                    for (; j < m; ++j)
                        Avx512bw::Copy(src + k + j * K, dst + j * 64 + k * 16);
                    for (; j < 16; ++j)
                        SetZero(dst + j * 64 + k * 16, _mm512_setzero_si512());
                }
                if (K64 < K)
                {
                    size_t j = 0;
                    for (; j < m; ++j)
                        Avx512bw::Copy(src + k + j * K, dst + j * 64 + k * 16, srcMask);
                    for (; j < 16; ++j)
                        SetZero(dst + j * 64 + k * 16, _mm512_setzero_si512());
                }
                src += K * 16;
                dst += a.aK * 16;
            }
        }

        //-------------------------------------------------------------------------------------------------

        static void QuantizedInnerProductGemmV1_Body2x2(const uint8_t* A0, const QipParam& p, const AlgParam& a, size_t M, size_t N, size_t K, int update, const int8_t* B0, int32_t* sum0)
        {
            int dS = (int)a.cN, dA = (int)a.bK, strideS = dS * 4, strideB = 64;
            int stepA = a.reorderType ? 1024 : 64, strideA = a.reorderType ? 64 : dA;
            const uint8_t* A1 = A0 + 16 * dA;
            const int8_t* B1 = B0 + a.bK * F;
            int32_t* sum1 = sum0 + 16 * dS;

            if (update)
            {
                _tile_stream_loadd(0, sum0 + 0, strideS);
                _tile_stream_loadd(1, sum0 + F, strideS);
                _tile_stream_loadd(2, sum1 + 0, strideS);
                _tile_stream_loadd(3, sum1 + F, strideS);
            }
            else
            {
                _tile_zero(0);
                _tile_zero(1);
                _tile_zero(2);
                _tile_zero(3);
            }

            int K64 = (int)K - 64, k = 0;
            _tile_stream_loadd(4, A0, strideA);
            _tile_loadd(6, B0 + k * 16, strideB);
            for (; k < K64; A1 += stepA)
            {
                _tile_loadd(7, B1 + k * 16, strideB);
                _tile_stream_loadd(5, A1, strideA);
                _tile_dpbusd(0, 4, 6);
                _tile_dpbusd(1, 4, 7);
                A0 += stepA;
                _tile_stream_loadd(4, A0, strideA);
                _tile_dpbusd(2, 5, 6);
                k += 64;
                _tile_loadd(6, B0 + k * 16, strideB);
                _tile_dpbusd(3, 5, 7);
            }
            _tile_loadd(7, B1 + k * 16, strideB);
            _tile_stream_loadd(5, A1, strideA);
            _tile_dpbusd(0, 4, 6);
            _tile_dpbusd(1, 4, 7);
            _tile_dpbusd(2, 5, 6);
            _tile_dpbusd(3, 5, 7);

            _tile_stored(0, sum0 + 0, strideS);
            TileMoveToMemory(sum0 + 0, dS);
            _tile_stored(1, sum0 + F, strideS);
            TileMoveToMemory(sum0 + F, dS);
            _tile_stored(2, sum1 + 0, strideS);
            TileMoveToMemory(sum1 + 0, dS);
            _tile_stored(3, sum1 + F, strideS);
            TileMoveToMemory(sum1 + F, dS);
        }

        static void QuantizedInnerProductGemmV1_Body2x1(const uint8_t* A0, const QipParam& p, const AlgParam& a, size_t M, size_t N, size_t K, int update, const int8_t* B0, int32_t* sum0)
        {
            int dS = (int)a.cN, dA = (int)a.bK, strideS = dS * 4, strideB = 64;
            int stepA = a.reorderType ? 1024 : 64, strideA = a.reorderType ? 64 : dA;
            const uint8_t* A1 = A0 + 16 * dA;
            int32_t* sum1 = sum0 + 16 * dS;

            if (update)
            {
                _tile_stream_loadd(0, sum0, strideS);
                _tile_stream_loadd(2, sum1, strideS);
            }
            else
            {
                _tile_zero(0);
                _tile_zero(2);
            }

            int K64 = (int)K - 64, k = 0;
            _tile_stream_loadd(4, A0, strideA);
            for (; k < K64; A1 += stepA, k += 64)
            {
                _tile_loadd(6, B0 + k * 16, strideB);
                _tile_stream_loadd(5, A1, strideA);
                _tile_dpbusd(0, 4, 6);
                A0 += stepA;
                _tile_stream_loadd(4, A0, strideA);
                _tile_dpbusd(2, 5, 6);
            }
            _tile_loadd(6, B0 + k * 16, strideB);
            _tile_stream_loadd(5, A1, strideA);
            _tile_dpbusd(0, 4, 6);
            _tile_dpbusd(2, 5, 6);

            _tile_stored(0, sum0, strideS);
            TileMoveToMemory(sum0, dS);
            _tile_stored(2, sum1, strideS);
            TileMoveToMemory(sum1, dS);
        }

        static void QuantizedInnerProductGemmV1_Body1x2(const uint8_t* A0, const QipParam& p, const AlgParam& a, size_t M, size_t N, size_t K, int update, const int8_t* B0, int32_t* sum)
        {
            int dS = (int)a.cN, dA = (int)a.bK, strideS = dS * 4, strideB = 64;
            int stepA = a.reorderType ? 1024 : 64, strideA = a.reorderType ? 64 : dA;
            const int8_t* B1 = B0 + a.bK * F;

            if (update)
            {
                _tile_stream_loadd(0, sum + 0, strideS);
                _tile_stream_loadd(1, sum + F, strideS);
            }
            else
            {
                _tile_zero(0);
                _tile_zero(1);
            }

            int K64 = (int)K - 64, k = 0;
            _tile_loadd(6, B0 + k * 16, strideB);
            for (; k < K64; A0 += stepA)
            {
                _tile_stream_loadd(4, A0, strideA);
                _tile_loadd(7, B1 + k * 16, strideB);
                _tile_dpbusd(0, 4, 6);
                k += 64;
                _tile_loadd(6, B0 + k * 16, strideB);
                _tile_dpbusd(1, 4, 7);
            }
            _tile_stream_loadd(4, A0, strideA);
            _tile_loadd(7, B1 + k * 16, strideB);
            _tile_dpbusd(0, 4, 6);
            _tile_dpbusd(1, 4, 7);

            _tile_stored(0, sum + 0, strideS);
            TileMoveToMemory(sum + 0, dS);
            _tile_stored(1, sum + F, strideS);
            TileMoveToMemory(sum + F, dS);
        }

        static void QuantizedInnerProductGemmV1_Body1x1(const uint8_t* A0, const QipParam& p, const AlgParam& a, size_t M, size_t N, size_t K, int update, const int8_t* B0, int32_t* sum)
        {
            int dS = (int)a.cN, dA = (int)a.bK, strideS = dS * 4, strideB = 64;
            int stepA = a.reorderType ? 1024 : 64, strideA = a.reorderType ? 64 : dA;

            if (update)
            {
                _tile_stream_loadd(0, sum + 0, strideS);
            }
            else
            {
                _tile_zero(0);
            }

            for (size_t k = 0; k < K; A0 += stepA, k += 64)
            {
                _tile_stream_loadd(4, A0, strideA);
                _tile_loadd(6, B0 + k * 16, strideB);
                _tile_dpbusd(0, 4, 6);
            }

            _tile_stored(0, sum + 0, strideS);
            TileMoveToMemory(sum + 0, dS);
        }

        typedef void(*QuantizedInnerProductGemmV1_BodyPtr)(const uint8_t* A0, const QipParam& p, const AlgParam& a, size_t M, size_t N, size_t K, int update, const int8_t* B0, int32_t* sum);

        static void QuantizedInnerProductGemmV1_Body(const uint8_t* A, const QipParam& p, const AlgParam& a, size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t* buf)
        {
            size_t n = 32;
            size_t Mn = AlignLoAny(M, n), m = M - Mn;
            size_t dB = a.cN, dA = a.bK;

            if (Mn)
            {
                bool avoidSrcOverflow = !(a.reorderType == 0 && p.K == a.aK);
                if (avoidSrcOverflow)
                    m = AlignHi(m, 16), Mn = M - m;
                QuantizedInnerProductGemmV1_BodyPtr body_2 = QuantizedInnerProductGemmV1_Body2x2;
                QuantizedInnerProductGemmV1_BodyPtr tail_2 = m > 16 ? QuantizedInnerProductGemmV1_Body2x2 : QuantizedInnerProductGemmV1_Body1x2;
                QuantizedInnerProductGemmV1_BodyPtr body_1 = QuantizedInnerProductGemmV1_Body2x1;
                QuantizedInnerProductGemmV1_BodyPtr tail_1 = m > 16 ? QuantizedInnerProductGemmV1_Body2x1 : QuantizedInnerProductGemmV1_Body1x1;
                SetTileConfFull();
                for (size_t j = 0; j < N; j += DF)
                {
                    size_t dN = Simd::Min(DF, N - j);
                    size_t i = 0;
                    if (dN > F)
                    {
                        for (; i < Mn; i += n)
                            body_2(A + i * dA, p, a, n, dN, K, update, B, buf + i * dB);
                        if (m)
                            tail_2(A + Mn * dA, p, a, m, dN, K, update, B, buf + i * dB);
                    }
                    else
                    {
                        for (; i < Mn; i += n)
                            body_1(A + i * dA, p, a, n, dN, K, update, B, buf + i * dB);
                        if (m)
                            tail_1(A + Mn * dA, p, a, m, dN, K, update, B, buf + i * dB);
                    }
                    B += a.bK * DF;
                    buf += DF;
                }
            }
            else
            {
                QuantizedInnerProductGemmV1_BodyPtr tail_2 = m > 16 ? QuantizedInnerProductGemmV1_Body2x2 : QuantizedInnerProductGemmV1_Body1x2;
                QuantizedInnerProductGemmV1_BodyPtr tail_1 = m > 16 ? QuantizedInnerProductGemmV1_Body2x1 : QuantizedInnerProductGemmV1_Body1x1;
                if (m > 16)
                    SetTileConf2x2(m, 32);
                else
                    SetTileConf1x2(m, 32);
                for (size_t j = 0; j < N; j += DF)
                {
                    size_t dN = Simd::Min(DF, N - j);
                    if (dN > F)
                        tail_2(A, p, a, m, dN, K, update, B, buf);
                    else
                        tail_1(A, p, a, m, dN, K, update, B, buf);
                    B += a.bK * DF;
                    buf += DF;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetQuantizedInnerProductGemmV1::SynetQuantizedInnerProductGemmV1(const QuantizedInnerProductParam& p)
            : Base::SynetQuantizedInnerProductGemmV1(p)
        {
            SetAlgParam();
            const AlgParam& a = _alg;
            if(a.reorderType)
                _prepA = QuantizedInnerProductGemmV1_PrepA_8uR;
            else
                _prepA = QuantizedInnerProductGemmV1_PrepA_8uD;
            _gemmBody = QuantizedInnerProductGemmV1_Body;
            //if (p.typeC == SimdTensorData8u)
            //    _gemm = QuantizedInnerProductGemmV0_2<Term8iLast8u>;
            //else
            //    _gemm = NULL;// QuantizedInnerProductGemmV0_2<Term8iLast32f>;
        }
    }
#endif
}
