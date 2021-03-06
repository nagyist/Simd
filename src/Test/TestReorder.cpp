/*
* Tests for Simd Library (http://simd.sourceforge.net).
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
#include "Test/TestUtils.h"
#include "Test/TestPerformance.h"
#include "Test/TestData.h"
#include "Test/Test.h"

namespace Test
{
	namespace
	{
		struct Func
		{
			typedef void (*FuncPtr)(const uint8_t * src, size_t size, uint8_t * dst);

			FuncPtr func;
			std::string description;

			Func(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & src, View & dst) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.width, dst.data);
			}
		};
	}

#define FUNC(function) Func(function, #function)

	bool ReorderAutoTest(int size, const Func & f1, const Func & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << size << "]." << std::endl;

		View s(size, 1, View::Gray8, NULL, TEST_ALIGN(size));
		FillRandom(s);

		View d1(size, 1, View::Gray8, NULL, TEST_ALIGN(size));
		View d2(size, 1, View::Gray8, NULL, TEST_ALIGN(size));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, d1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, d2));

		result = result && Compare(d1, d2, 0, true, 64);

		return result;
	}

    bool ReorderAutoTest(const Func & f1, const Func & f2, int bytes)
    {
        bool result = true;

        result = result && ReorderAutoTest(W*H, f1, f2);
        result = result && ReorderAutoTest(W*H + O*bytes, f1, f2);
        result = result && ReorderAutoTest(W*H - O*bytes, f1, f2);

        return result;
    }

	bool Reorder16bitAutoTest()
	{
		bool result = true;

        result = result && ReorderAutoTest(FUNC(Simd::Base::Reorder16bit), FUNC(SimdReorder16bit), 2);

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Sse2::Reorder16bit), FUNC(SimdReorder16bit), 2);
#endif 

#ifdef SIMD_SSSE3_ENABLE
        if(Simd::Ssse3::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Ssse3::Reorder16bit), FUNC(SimdReorder16bit), 2);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Avx2::Reorder16bit), FUNC(SimdReorder16bit), 2);
#endif

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Vsx::Reorder16bit), FUNC(SimdReorder16bit), 2);
#endif 

		return result;
	}

    bool Reorder32bitAutoTest()
    {
        bool result = true;

        result = result && ReorderAutoTest(FUNC(Simd::Base::Reorder32bit), FUNC(SimdReorder32bit), 4);

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Sse2::Reorder32bit), FUNC(SimdReorder32bit), 4);
#endif 

#ifdef SIMD_SSSE3_ENABLE
        if(Simd::Ssse3::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Ssse3::Reorder32bit), FUNC(SimdReorder32bit), 4);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Avx2::Reorder32bit), FUNC(SimdReorder32bit), 4);
#endif

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Vsx::Reorder32bit), FUNC(SimdReorder32bit), 4);
#endif 

        return result;
    }

    bool Reorder64bitAutoTest()
    {
        bool result = true;

        result = result && ReorderAutoTest(FUNC(Simd::Base::Reorder64bit), FUNC(SimdReorder64bit), 8);

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Sse2::Reorder64bit), FUNC(SimdReorder64bit), 8);
#endif 

#ifdef SIMD_SSSE3_ENABLE
        if(Simd::Ssse3::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Ssse3::Reorder64bit), FUNC(SimdReorder64bit), 8);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Avx2::Reorder64bit), FUNC(SimdReorder64bit), 8);
#endif

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Vsx::Reorder64bit), FUNC(SimdReorder64bit), 8);
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool ReorderDataTest(bool create, int size, const Func & f)
    {
        bool result = true;

        Data data(f.description);

        std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << size << "]." << std::endl;

        View s(size, 1, View::Gray8, NULL, TEST_ALIGN(size));
        View d1(size, 1, View::Gray8, NULL, TEST_ALIGN(size));
        View d2(size, 1, View::Gray8, NULL, TEST_ALIGN(size));

        if(create)
        {
            FillRandom(s);
            TEST_SAVE(s);

            f.Call(s, d1);

            TEST_SAVE(d1);
        }
        else
        {
            TEST_LOAD(s);
            TEST_LOAD(d1);

            f.Call(s, d2);

            TEST_SAVE(d2);

            result = result && Compare(d1, d2, 0, true, 64);
        }

        return result;
    }

    bool Reorder16bitDataTest(bool create)
    {
        bool result = true;

        result = result && ReorderDataTest(create, DS, FUNC(SimdReorder16bit));

        return result;
    }

    bool Reorder32bitDataTest(bool create)
    {
        bool result = true;

        result = result && ReorderDataTest(create, DS, FUNC(SimdReorder32bit));

        return result;
    }

    bool Reorder64bitDataTest(bool create)
    {
        bool result = true;

        result = result && ReorderDataTest(create, DS, FUNC(SimdReorder64bit));

        return result;
    }
}