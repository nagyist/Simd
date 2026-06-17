/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#ifndef __SimdPack_h__
#define __SimdPack_h__

#include "Simd/SimdDefs.h"
#include "Simd/SimdCast.h"

namespace Simd
{
    namespace Base
    {
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
    }
#endif   

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
    }
#endif  

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
    }
#endif 

#ifdef SIMD_AMXBF16_ENABLE    
    namespace AmxBf16
    {
    }
#endif

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {

    }
#endif 

#ifdef SIMD_SVE2_ENABLE    
    namespace Sve2
    {
        SIMD_INLINE svuint8_t PackSeqI16ToU8(const svint16_t& lo, const svint16_t& hi)
        {
            return svuzp1_u8(To8u(lo), To8u(hi));
        }

        SIMD_INLINE svuint8_t PackSatIntI16ToU8(const svint16_t& even, const svint16_t& odd)
        {
            return svqxtunt_s16(svqxtunb_s16(even), odd);
        }

        SIMD_INLINE svint16_t PackSatIntI32ToI16(const svint32_t& lo, const svint32_t& hi)
        {
            return svqxtnt_s32(svqxtnb_s32(lo), hi);
        }
    }
#endif 
}

#endif
