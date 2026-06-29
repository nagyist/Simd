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

        template<Term8iType term, int flush, int M> static SIMD_INLINE void ApplyMx1(
            uint8_t* ptr, int32_t* buf, const __m512i* sBias, const __m512* sNorm, const __m512i& dZero, __mmask32 tail = -1)
        {
            __m512i d0, d1;
            if (M > 0) d0 = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_loadu_si512(buf + 0), sBias[0])), sNorm[0])), dZero);
            if (M > 1) d1 = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_loadu_si512(buf + F), sBias[1])), sNorm[1])), dZero);
            if (term == Term8iLast8u)
            {
                if (M > 1)
                {
                    _mm256_mask_storeu_epi8(ptr, tail, _mm512_castsi512_si256(PackI16ToU8(PackI32ToI16(d0, d1), K_ZERO)));
                    if (flush) _mm_prefetch((const char*)ptr, _MM_HINT_NTA);
                }
                else
                {
                    _mm_mask_storeu_epi8(ptr, tail, _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(d0, K_ZERO), K_ZERO)));
                    if (flush) _mm_prefetch((const char*)ptr, _MM_HINT_NTA);
                }
            }
            else if (term == Term8iLast32f)
            {
                assert(0);
            }
        }
        template<Term8iType term, int flush, int M, int N> static SIMD_INLINE void ApplyMxN(
            uint8_t* ptr, size_t dP, int32_t* buf, const __m512i* sBias, const __m512* sNorm, const __m512i& dZero, __mmask32 tail = -1)
        {
            if (N > 0) ApplyMx1<term, flush, M>(ptr + 0 * dP, buf + 0 * DF, sBias, sNorm, dZero, tail);
            if (N > 1) ApplyMx1<term, flush, M>(ptr + 1 * dP, buf + 1 * DF, sBias, sNorm, dZero, tail);
            if (N > 2) ApplyMx1<term, flush, M>(ptr + 2 * dP, buf + 2 * DF, sBias, sNorm, dZero, tail);
            if (N > 3) ApplyMx1<term, flush, M>(ptr + 3 * dP, buf + 3 * DF, sBias, sNorm, dZero, tail);
            if (N > 4) ApplyMx1<term, flush, M>(ptr + 4 * dP, buf + 4 * DF, sBias, sNorm, dZero, tail);
            if (N > 5) ApplyMx1<term, flush, M>(ptr + 5 * dP, buf + 5 * DF, sBias, sNorm, dZero, tail);
            if (N > 6) ApplyMx1<term, flush, M>(ptr + 6 * dP, buf + 6 * DF, sBias, sNorm, dZero, tail);
            if (N > 7) ApplyMx1<term, flush, M>(ptr + 7 * dP, buf + 7 * DF, sBias, sNorm, dZero, tail);
        }

        //--------------------------------------------------------------------------------------------------

        template<Term8iType term, int flush, int N, int apply> SIMD_INLINE void QuantizedInnerProductGemmV1_Last1xMx2(
            const uint8_t* A0, const QipParam& p, const AlgParam& a, size_t K, int update, const int8_t* B0, const __m512i* sBias, const __m512* sNorm,
            const __m512i& dZero, int32_t* sum0, int32_t* buf1, int32_t* buf2, uint8_t* C, __mmask32 tailN)
        {
            int dC = int(p.N * a.eC), dA = (int)a.bK, dB = 16, strideB = 64;
            int stepA = a.reorderType ? 1024 : 64, strideA = a.reorderType ? 64 : dA;
            const uint8_t* A1 = A0 + 16 * dA;
            const int8_t* B1 = B0 + a.bK * F;

            if (update)
            {
                int dS = (int)a.cN, strideS = dS * 4;
                if (N > 0) _tile_stream_loadd(0, sum0 + 0, strideS);
                if (N > 1) _tile_stream_loadd(1, sum0 + F, strideS);
                sum0 += 16 * dS;
                if (N > 0) _tile_stream_loadd(2, sum0 + 0, strideS);
                if (N > 1) _tile_stream_loadd(3, sum0 + F, strideS);
            }
            else
            {
                if (N > 0) _tile_zero(0);
                if (N > 1) _tile_zero(1);
                if (N > 0) _tile_zero(2);
                if (N > 1) _tile_zero(3);
            }

            int K64 = (int)K - 64, aK64 = apply ? (8 * 64 / apply - 64) : 0, k = 0, i = 0;

            _tile_stream_loadd(4, A0, strideA);
            if (N > 0) _tile_loadd(6, B0 + k * dB, strideB);
            for (; k < aK64; A1 += stepA)
            {
                if (N > 1) _tile_loadd(7, B1 + k * dB, strideB);
                if (N > 0) _tile_dpbusd(0, 4, 6);
                ApplyMxN<term, flush, N, apply>(C + i * dC, dC, buf1 + i * DF, sBias, sNorm, dZero, tailN), i += apply;
                _tile_stream_loadd(5, A1, strideA);
                if (N > 1) _tile_dpbusd(1, 4, 7);
                ApplyMxN<term, flush, N, apply>(C + i * dC, dC, buf1 + i * DF, sBias, sNorm, dZero, tailN), i += apply;
                A0 += stepA;
                _tile_stream_loadd(4, A0, strideA);
                if (N > 0) _tile_dpbusd(2, 5, 6);
                ApplyMxN<term, flush, N, apply>(C + i * dC, dC, buf1 + i * DF, sBias, sNorm, dZero, tailN), i += apply;
                k += 64;
                if (N > 0) _tile_loadd(6, B0 + k * dB, strideB);
                if (N > 1) _tile_dpbusd(3, 5, 7);
                ApplyMxN<term, flush, N, apply>(C + i * dC, dC, buf1 + i * DF, sBias, sNorm, dZero, tailN), i += apply;
            }
            for (; k < K64; A1 += stepA)
            {
                if (N > 1) _tile_loadd(7, B1 + k * dB, strideB);
                if (N > 0) _tile_dpbusd(0, 4, 6);
                _tile_stream_loadd(5, A1, strideA);
                if (N > 1) _tile_dpbusd(1, 4, 7);
                A0 += stepA;
                _tile_stream_loadd(4, A0, strideA);
                if (N > 0) _tile_dpbusd(2, 5, 6);
                k += 64;
                if (N > 0) _tile_loadd(6, B0 + k * dB, strideB);
                if (N > 1) _tile_dpbusd(3, 5, 7);
            }
            if (N > 1) _tile_loadd(7, B1 + k * dB, strideB);
            _tile_stream_loadd(5, A1, strideA);
            if (N > 0) _tile_dpbusd(0, 4, 6);
            if (apply) ApplyMxN<term, flush, N, apply>(C + i * dC, dC, buf1 + i * DF, sBias, sNorm, dZero, tailN), i += apply;
            if (N > 0) _tile_stored(0, buf2 + 0, DA);
            if (N > 1) _tile_dpbusd(1, 4, 7);
            if (apply) ApplyMxN<term, flush, N, apply>(C + i * dC, dC, buf1 + i * DF, sBias, sNorm, dZero, tailN), i += apply;
            if (N > 1) _tile_stored(1, buf2 + F, DA);
            buf2 += 16 * DF;
            if (N > 0) _tile_dpbusd(2, 5, 6);
            if (apply) ApplyMxN<term, flush, N, apply>(C + i * dC, dC, buf1 + i * DF, sBias, sNorm, dZero, tailN), i += apply;
            if (N > 0) _tile_stored(2, buf2 + 0, DA);
            if (N > 1) _tile_dpbusd(3, 5, 7);
            if (apply) ApplyMxN<term, flush, N, apply>(C + i * dC, dC, buf1 + i * DF, sBias, sNorm, dZero, tailN), i += apply;
            if (N > 1) _tile_stored(3, buf2 + F, DA);
        }

        template<Term8iType term, int flush, int N, int apply> SIMD_INLINE void QuantizedInnerProductGemmV1_Last1xMx1(
            const uint8_t* A0, const QipParam& p, const AlgParam& a, size_t K, int update, const int8_t* B0, const __m512i* sBias, const __m512* sNorm,
            const __m512i& dZero, int32_t* sum0, int32_t* buf1, int32_t* buf2, uint8_t* C, __mmask32 tailN)
        {
            int dC = int(p.N * a.eC), dA = (int)a.bK, dB = 16, strideB = 64;
            int stepA = a.reorderType ? 1024 : 64, strideA = a.reorderType ? 64 : dA;
            const uint8_t* A1 = A0 + 16 * dA;
            const int8_t* B1 = B0 + a.bK * F;

            if (update)
            {
                int dS = (int)a.cN, strideS = dS * 4;
                if (N > 0) _tile_stream_loadd(0, sum0 + 0, strideS);
                if (N > 1) _tile_stream_loadd(1, sum0 + F, strideS);
            }
            else
            {
                if (N > 0) _tile_zero(0);
                if (N > 1) _tile_zero(1);
            }

            int K64 = (int)K - 64, aK64 = apply ? (8 * 64 / apply - 64) : 0, k = 0, i = 0;

            _tile_stream_loadd(4, A0, strideA);
            if (N > 0) _tile_loadd(6, B0 + k * dB, strideB);
            for (; k < aK64;)
            {
                if (N > 1) _tile_loadd(7, B1 + k * dB, strideB);
                if (N > 0) _tile_dpbusd(0, 4, 6);
                ApplyMxN<term, flush, N, apply>(C + i * dC, dC, buf1 + i * DF, sBias, sNorm, dZero, tailN), i += apply;
                ApplyMxN<term, flush, N, apply>(C + i * dC, dC, buf1 + i * DF, sBias, sNorm, dZero, tailN), i += apply;
                k += 64;
                if (N > 0) _tile_loadd(6, B0 + k * dB, strideB);
                if (N > 1) _tile_dpbusd(1, 4, 7);
                ApplyMxN<term, flush, N, apply>(C + i * dC, dC, buf1 + i * DF, sBias, sNorm, dZero, tailN), i += apply;
                ApplyMxN<term, flush, N, apply>(C + i * dC, dC, buf1 + i * DF, sBias, sNorm, dZero, tailN), i += apply;
                A0 += stepA;
                _tile_stream_loadd(4, A0, strideA);
            }
            for (; k < K64;)
            {
                if (N > 1) _tile_loadd(7, B1 + k * dB, strideB);
                if (N > 0) _tile_dpbusd(0, 4, 6);
                k += 64;
                if (N > 0) _tile_loadd(6, B0 + k * dB, strideB);
                if (N > 1) _tile_dpbusd(1, 4, 7);
                A0 += stepA;
                _tile_stream_loadd(4, A0, strideA);
            }
            if (N > 1) _tile_loadd(7, B1 + k * dB, strideB);
            if (N > 0) _tile_dpbusd(0, 4, 6);
            if (apply) ApplyMxN<term, flush, N, apply>(C + i * dC, dC, buf1 + i * DF, sBias, sNorm, dZero, tailN), i += apply;
            if (apply) ApplyMxN<term, flush, N, apply>(C + i * dC, dC, buf1 + i * DF, sBias, sNorm, dZero, tailN), i += apply;
            if (N > 0) _tile_stored(0, buf2 + 0, DA);
            if (N > 1) _tile_dpbusd(1, 4, 7);
            if (apply) ApplyMxN<term, flush, N, apply>(C + i * dC, dC, buf1 + i * DF, sBias, sNorm, dZero, tailN), i += apply;
            if (apply) ApplyMxN<term, flush, N, apply>(C + i * dC, dC, buf1 + i * DF, sBias, sNorm, dZero, tailN), i += apply;
            if (N > 1) _tile_stored(1, buf2 + F, DA);
        }

        template<Term8iType term, int flush, int N, int apply> void QuantizedInnerProductGemmV1_LastNxMx2(const uint8_t* A0, const QipParam& p,
            const AlgParam& a, size_t M, size_t K, int update, const int8_t* B0, const __m512i* sBias, const __m512* sNorm,
            const __m512i& dZero, int32_t* sum0, int32_t* buf1, uint8_t* C, __mmask32 tailN)
        {
            int dS = (int)a.cN, dC = int(p.N * a.eC), dA = (int)a.bK;
            int32_t* buf2 = buf1 + 1024;

            size_t ci = 0, pi = 0;
            QuantizedInnerProductGemmV1_Last1xMx2<term, flush, N, 0>(A0, p, a, K, update, B0,
                sBias, sNorm, dZero, sum0, NULL, buf2, C, tailN), ci += 32;
            for (; ci < M; pi = ci, ci += 32)
            {
                Swap(buf1, buf2);
                size_t si = ci;
                if (ci + 16 >= M)
                {
                    if (a.reorderType == 0)
                        ci = Simd::Min(M - 16, ci);
                    QuantizedInnerProductGemmV1_Last1xMx1<term, flush, N, apply>(A0 + ci * dA, p, a, K, update, B0,
                        sBias, sNorm, dZero, sum0 + si * dS, buf1, buf2, C + pi * dC, tailN);
                }
                else
                {
                    if (a.reorderType == 0)
                        ci = Simd::Min(M - 32, ci);
                    QuantizedInnerProductGemmV1_Last1xMx2<term, flush, N, apply>(A0 + ci * dA, p, a, K, update, B0,
                        sBias, sNorm, dZero, sum0 + si * dS, buf1, buf2, C + pi * dC, tailN);
                }
            }
            uint8_t* C1 = C + pi * dC;
            M -= pi;
            size_t i = 0, M8 = M & (~7);
            for (; i < M8; i += 8)
                ApplyMxN<term, flush, N, 8>(C1 + i * dC, dC, buf2 + i * DF, sBias, sNorm, dZero, tailN);
            for (; i < M; ++i)
                ApplyMxN<term, flush, N, 1>(C1 + i * dC, dC, buf2 + i * DF, sBias, sNorm, dZero, tailN);
        }

        //--------------------------------------------------------------------------------------------------

        template<Term8iType term, int flush> void QuantizedInnerProductGemmV1_Last2x2(const uint8_t* A0, const QipParam& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const int8_t* B0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* sum0, int32_t* buf0, uint8_t* C, __mmask32 tailN)
        {
            int dS = (int)a.cN, dC = int(p.N * a.eC), dA = (int)a.bK, strideS = dS * 4, strideB = 64;
            int stepA = a.reorderType ? 1024 : 64, strideA = a.reorderType ? 64 : dA;
            const uint8_t* A1 = A0 + 16 * dA;
            const int8_t* B1 = B0 + a.bK * F;
            int32_t* sum1 = sum0 + 16 * dS;
            int32_t* buf1 = buf0 + 16 * DF;

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

            _tile_stored(0, buf0 + 0, DA);
            _tile_stored(1, buf0 + F, DA);
            _tile_stored(2, buf1 + 0, DA);
            _tile_stored(3, buf1 + F, DA);
            if (term == Term8iLast8u)
            {
                size_t M8 = M & (~7), i = 0;
                for (; i < M8; i += 8)
                    ApplyMxN<term, flush, 2, 8>(C + i * dC, dC, buf0 + i * DF, bias, norm, zero, tailN);
                for (; i < M; ++i)
                    ApplyMxN<term, flush, 2, 1>(C + i * dC, dC, buf0 + i * DF, bias, norm, zero, tailN);
            }
            else if (term == Term8iLast32f)
            {
            }
        }

        template<Term8iType term, int flush> void QuantizedInnerProductGemmV1_Last2x1(const uint8_t* A0, const QipParam& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const int8_t* B0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* sum0, int32_t* buf0, uint8_t* C, __mmask32 tailN)
        {
            int dS = (int)a.cN, dC = int(p.N * a.eC), dA = (int)a.bK, strideS = dS * 4, strideB = 64;
            int stepA = a.reorderType ? 1024 : 64, strideA = a.reorderType ? 64 : dA;
            const uint8_t* A1 = A0 + 16 * dA;
            int32_t* sum1 = sum0 + 16 * dS;
            int32_t* buf1 = buf0 + 16 * DF;

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

            _tile_stored(0, buf0 + 0, DA);
            _tile_stored(2, buf1 + 0, DA);
            if (term == Term8iLast8u)
            {
                size_t M8 = M & (~7), i = 0;
                for (; i < M8; i += 8)
                    ApplyMxN<term, flush, 1, 8>(C + i * dC, dC, buf0 + i * DF, bias, norm, zero, tailN);
                for (; i < M; ++i)
                    ApplyMxN<term, flush, 1, 1>(C + i * dC, dC, buf0 + i * DF, bias, norm, zero, tailN);
            }
            else if (term == Term8iLast32f)
            {
            }
        }

        template<Term8iType term, int flush> void QuantizedInnerProductGemmV1_Last1x2(const uint8_t* A0, const QipParam& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const int8_t* B0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* sum, int32_t* buf0, uint8_t* C, __mmask32 tailN)
        {
            int dS = (int)a.cN, dC = int(p.N * a.eC), dA = (int)a.bK, strideS = dS * 4, strideB = 64;
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

            _tile_stored(0, buf0 + 0, DA);
            _tile_stored(1, buf0 + F, DA);
            if (term == Term8iLast8u)
            {
                size_t M8 = M & (~7), i = 0;
                for (; i < M8; i += 8)
                    ApplyMxN<term, flush, 2, 8>(C + i * dC, dC, buf0 + i * DF, bias, norm, zero, tailN);
                for (; i < M; ++i)
                    ApplyMxN<term, flush, 2, 1>(C + i * dC, dC, buf0 + i * DF, bias, norm, zero, tailN);
            }
            else if (term == Term8iLast32f)
            {
            }
        }

        template<Term8iType term, int flush> void QuantizedInnerProductGemmV1_Last1x1(const uint8_t* A0, const QipParam& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const int8_t* B0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* sum, int32_t* buf0, uint8_t* C, __mmask32 tailN)
        {
            int dS = (int)a.cN, dC = int(p.N * a.eC), dA = (int)a.bK, strideS = dS * 4, strideB = 64;
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

            _tile_stored(0, buf0 + 0, DA);
            if (term == Term8iLast8u)
            {
                size_t M8 = M & (~7), i = 0;
                for (; i < M8; i += 8)
                    ApplyMxN<term, flush, 1, 8>(C + i * dC, dC, buf0 + i * DF, bias, norm, zero, tailN);
                for (; i < M; ++i)
                    ApplyMxN<term, flush, 1, 1>(C + i * dC, dC, buf0 + i * DF, bias, norm, zero, tailN);
            }
            else if (term == Term8iLast32f)
            {
            }
        }

        typedef void(*QuantizedInnerProductGemmV1_LastPtr)(const uint8_t* A0, const QipParam& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const int8_t* B0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* sum, int32_t* buf, uint8_t* C, __mmask32 tailN);

        //--------------------------------------------------------------------------------------------------

        template<Term8iType term, int flush, int apply> void QuantizedInnerProductGemmV1_Last(const uint8_t* A, const QipParam& p, const AlgParam& a, size_t M, size_t N, size_t K,
            int update, const int8_t* B, int32_t* sum, int32_t* buf, const int32_t* bias, const float* norm, uint32_t zero, uint8_t* C)
        {
            size_t n = 32;
            size_t Mn = AlignLoAny(M, n), m = M - Mn;
            size_t dB = a.cN, dC = p.N * a.eC, dA = a.bK;
            __m512 _norm[2];
            __m512i _bias[2], _zero = _mm512_set1_epi32(zero);
            if (Mn)
            {
                bool avoidSrcOverflow = !(a.reorderType == 0 && p.K == a.aK);
                if (avoidSrcOverflow)
                    m = AlignHi(m, 16), Mn = M - m;
                QuantizedInnerProductGemmV1_LastPtr body_2 = QuantizedInnerProductGemmV1_Last2x2<term, flush>;
                QuantizedInnerProductGemmV1_LastPtr tail_2 = m > 16 ? QuantizedInnerProductGemmV1_Last2x2<term, flush> : QuantizedInnerProductGemmV1_Last1x2<term, flush>;
                QuantizedInnerProductGemmV1_LastPtr body_1 = QuantizedInnerProductGemmV1_Last2x1<term, flush>;
                QuantizedInnerProductGemmV1_LastPtr tail_1 = m > 16 ? QuantizedInnerProductGemmV1_Last2x1<term, flush> : QuantizedInnerProductGemmV1_Last1x1<term, flush>;
                SetTileConfFull();
                for (size_t j = 0; j < N; j += DF)
                {
                    size_t dN = Simd::Min(DF, N - j);
                    _bias[0] = _mm512_loadu_si512((__m512i*)(bias + j) + 0);
                    _bias[1] = _mm512_loadu_si512((__m512i*)(bias + j) + 1);
                    _norm[0] = _mm512_loadu_ps(norm + j + 0);
                    _norm[1] = _mm512_loadu_ps(norm + j + F);
                    size_t i = 0;
                    __mmask32 tailN = TailMask32(dN);
                    if (dN > F)
                        QuantizedInnerProductGemmV1_LastNxMx2<term, flush, 2, apply>(A + i * dA, p, a, M, K, update, B, _bias, _norm, _zero, sum + i * dB, buf, C + i * dC, tailN);
                    else
                        QuantizedInnerProductGemmV1_LastNxMx2<term, flush, 1, apply>(A + i * dA, p, a, M, K, update, B, _bias, _norm, _zero, sum + i * dB, buf, C + i * dC, tailN);
                    B += a.bK * DF;
                    sum += DF;
                    C += DF * a.eC;
                }
            }
            else
            {
                QuantizedInnerProductGemmV1_LastPtr tail_2 = m > 16 ? QuantizedInnerProductGemmV1_Last2x2<term, flush> : QuantizedInnerProductGemmV1_Last1x2<term, flush>;
                QuantizedInnerProductGemmV1_LastPtr tail_1 = m > 16 ? QuantizedInnerProductGemmV1_Last2x1<term, flush> : QuantizedInnerProductGemmV1_Last1x1<term, flush>;
                if (m > 16)
                    SetTileConf2x2(m, 32);
                else
                    SetTileConf1x2(m, 32);
                for (size_t j = 0; j < N; j += DF)
                {
                    size_t dN = Simd::Min(DF, N - j);
                    _bias[0] = _mm512_loadu_si512((__m512i*)(bias + j) + 0);
                    _bias[1] = _mm512_loadu_si512((__m512i*)(bias + j) + 1);
                    _norm[0] = _mm512_loadu_ps(norm + j + 0);
                    _norm[1] = _mm512_loadu_ps(norm + j + F);
                    __mmask32 tailN = TailMask32(dN);
                    if (dN > F)
                        tail_2(A, p, a, m, dN, K, update, B, _bias, _norm, _zero, sum, buf, C, tailN);
                    else
                        tail_1(A, p, a, m, dN, K, update, B, _bias, _norm, _zero, sum, buf, C, tailN);
                    B += a.bK * DF;
                    sum += DF;
                    C += DF * a.eC;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetQuantizedInnerProductGemmV1::SynetQuantizedInnerProductGemmV1(const QuantizedInnerProductParam& p)
            : Base::SynetQuantizedInnerProductGemmV1(p)
        {
            SetAlgParam();
            const AlgParam& a = _alg;
            if (_sizeA)
            {
                if (a.reorderType)
                    _prepA = QuantizedInnerProductGemmV1_PrepA_8uR;
                else
                    _prepA = QuantizedInnerProductGemmV1_PrepA_8uD;
            }
            _gemmBody = QuantizedInnerProductGemmV1_Body;
            if (p.typeC == SimdTensorData8u)
            {
                size_t K = p.K - AlignLoAny(p.K, a.macroK);
                if (K > 448)
                    _gemmLast = QuantizedInnerProductGemmV1_Last<Term8iLast8u, 0, 1>;
                else if (K > 192)
                    _gemmLast = QuantizedInnerProductGemmV1_Last<Term8iLast8u, 0, 2>;
                else if (K > 64)
                    _gemmLast = QuantizedInnerProductGemmV1_Last<Term8iLast8u, 0, 4>;
                else
                    _gemmLast = QuantizedInnerProductGemmV1_Last<Term8iLast8u, 0, 8>;
            }
            else
                _gemmLast = NULL;
        }
    }
#endif
}
