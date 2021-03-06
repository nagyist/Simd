/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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
#include "Simd/SimdVsx.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_VSX_ENABLE  
    namespace Vsx
    {
        namespace
        {
            struct Buffer
            {
                Buffer(size_t count)
                {
                    size_t size = sizeof(float)*count;
                    _p = Allocate(size);
                    memset(_p, 0, size);
                    sums = (float*)_p;
                }

                ~Buffer()
                {
                    Free(_p);
                }

                float * sums;
            private:
                void *_p;
            };
        }

        void SvmSumLinear(const float * x, const float * svs, const float * weights, size_t length, size_t count, float * sum)
        {
            Buffer buffer(count);
            size_t alignedCount = AlignLo(count, 4);

            for(size_t j = 0; j < length; ++j)
            {
                size_t i = 0;
                float v = x[j];
                v128_f32 _v = SetF32(v);
                for(; i < alignedCount; i += 4)
                {
                    v128_f32 sums = Load<true>(buffer.sums + i);
                    v128_f32 _svs = Load<false>(svs + i);
                    Store<true>(buffer.sums + i, vec_add(sums, vec_mul(_v, _svs)));
                }
                for(; i < count; ++i)
                    buffer.sums[i] += v*svs[i];
                svs += count;
            }

            size_t i = 0;
            v128_f32 _sum = K_0_0f;
            for(; i < alignedCount; i += 4)
            {
                v128_f32 sums = Load<true>(buffer.sums + i);
                v128_f32 _weights = Load<false>(weights + i);
                _sum = vec_add(_sum, vec_mul(sums, _weights));
            }
            *sum = ExtractSum(_sum);
            for(; i < count; ++i)
                *sum += buffer.sums[i]*weights[i];
        }
    }
#endif// SIMD_VSX_ENABLE
}