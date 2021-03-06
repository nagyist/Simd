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
		struct FuncOB8U
		{
			typedef void (*FuncPtr)(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
				size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride, SimdOperationBinary8uType type);

			FuncPtr func;
			std::string description;

			FuncOB8U(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & a, const View & b, View & dst, SimdOperationBinary8uType type) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(a.data, a.stride, b.data, b.stride, a.width, a.height, View::PixelSize(a.format), dst.data, dst.stride, type);
			}
		};

        struct FuncOB16I
        {
            typedef void (*FuncPtr)(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
                size_t width, size_t height, uint8_t * dst, size_t dstStride, SimdOperationBinary16iType type);

            FuncPtr func;
            std::string description;

            FuncOB16I(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

            void Call(const View & a, const View & b, View & dst, SimdOperationBinary16iType type) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(a.data, a.stride, b.data, b.stride, a.width, a.height, dst.data, dst.stride, type);
            }
        };

        struct FuncVP
        {
            typedef void (*FuncPtr)(const uint8_t * vertical, const uint8_t * horizontal, uint8_t * dst, size_t stride, size_t width, size_t height);

            FuncPtr func;
            std::string description;

            FuncVP(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

            void Call(const View & v, const View & h, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(v.data, h.data, dst.data, dst.stride, dst.width, dst.height);
            }
        };
	}

    SIMD_INLINE std::string OperationBinary8uTypeDescription(SimdOperationBinary8uType type)
    {
        switch(type)
        {
        case SimdOperationBinary8uAverage:
            return "<Avg>";
        case SimdOperationBinary8uAnd:
            return "<And>";
        case SimdOperationBinary8uMaximum:
            return "<Max>";
        case SimdOperationBinary8uSaturatedSubtraction:
            return "<Subs>";
        case SimdOperationBinary8uSaturatedAddition:
            return "<Adds>";
        }
		assert(0);
		return "<Unknown";
    }

    SIMD_INLINE std::string OperationBinary16iTypeDescription(SimdOperationBinary16iType type)
    {
        switch(type)
        {
        case SimdOperationBinary16iAddition:
            return "<Add>";
        }
        assert(0);
        return "<Unknown";
    }

#define ARGS_OB8U(format, width, height, type, function1, function2) \
	format, width, height, type, \
	FuncOB8U(function1.func, function1.description + OperationBinary8uTypeDescription(type) + ColorDescription(format)), \
	FuncOB8U(function2.func, function2.description + OperationBinary8uTypeDescription(type) + ColorDescription(format))

#define FUNC_OB8U(function) \
	FuncOB8U(function, std::string(#function))

#define ARGS_OB16I(width, height, type, function1, function2) \
    width, height, type, \
    FuncOB16I(function1.func, function1.description + OperationBinary16iTypeDescription(type)), \
    FuncOB16I(function2.func, function2.description + OperationBinary16iTypeDescription(type))

#define FUNC_OB16I(function) \
    FuncOB16I(function, std::string(#function))

#define ARGS_VP(function) FuncVP(function, std::string(#function))

	bool OperationBinary8uAutoTest(View::Format format, int width, int height, SimdOperationBinary8uType type, const FuncOB8U & f1, const FuncOB8U & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View a(width, height, format, NULL, TEST_ALIGN(width));
		FillRandom(a);

		View b(width, height, format, NULL, TEST_ALIGN(width));
		FillRandom(b);

		View d1(width, height, format, NULL, TEST_ALIGN(width));
		View d2(width, height, format, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(a, b, d1, type));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(a, b, d2, type));

		result = result && Compare(d1, d2, 0, true, 64);

		return result;
	}

	bool OperationBinary8uAutoTest(const FuncOB8U & f1, const FuncOB8U & f2)
	{
		bool result = true;

        for(SimdOperationBinary8uType type = SimdOperationBinary8uAverage; type <= SimdOperationBinary8uSaturatedAddition && result; type = SimdOperationBinary8uType(type + 1))
        {
            for(View::Format format = View::Gray8; format <= View::Bgra32; format = View::Format(format + 1))
            {
                result = result && OperationBinary8uAutoTest(ARGS_OB8U(format, W, H, type, f1, f2));
                result = result && OperationBinary8uAutoTest(ARGS_OB8U(format, W + O, H - O, type, f1, f2));
                result = result && OperationBinary8uAutoTest(ARGS_OB8U(format, W - O, H + O, type, f1, f2));
            }
        }

		return result;
	}

	bool OperationBinary8uAutoTest()
	{
		bool result = true;

		result = result && OperationBinary8uAutoTest(FUNC_OB8U(Simd::Base::OperationBinary8u), FUNC_OB8U(SimdOperationBinary8u));

#ifdef SIMD_SSE2_ENABLE
		if(Simd::Sse2::Enable)
			result = result && OperationBinary8uAutoTest(FUNC_OB8U(Simd::Sse2::OperationBinary8u), FUNC_OB8U(SimdOperationBinary8u));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && OperationBinary8uAutoTest(FUNC_OB8U(Simd::Avx2::OperationBinary8u), FUNC_OB8U(SimdOperationBinary8u));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && OperationBinary8uAutoTest(FUNC_OB8U(Simd::Vsx::OperationBinary8u), FUNC_OB8U(SimdOperationBinary8u));
#endif 

		return result;
	}

    bool OperationBinary16iAutoTest(int width, int height, SimdOperationBinary16iType type, const FuncOB16I & f1, const FuncOB16I & f2)
    {
        bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

        View a(width, height, View::Int16, NULL, TEST_ALIGN(width));
        FillRandom(a);

        View b(width, height, View::Int16, NULL, TEST_ALIGN(width));
        FillRandom(b);

        View d1(width, height, View::Int16, NULL, TEST_ALIGN(width));
        View d2(width, height, View::Int16, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(a, b, d1, type));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(a, b, d2, type));

        result = result && Compare(d1, d2, 0, true, 64);

        return result;
    }

    bool OperationBinary16iAutoTest(const FuncOB16I & f1, const FuncOB16I & f2)
    {
        bool result = true;

        for(SimdOperationBinary16iType type = SimdOperationBinary16iAddition; type <= SimdOperationBinary16iAddition && result; type = SimdOperationBinary16iType(type + 1))
        {
            result = result && OperationBinary16iAutoTest(ARGS_OB16I(W, H, type, f1, f2));
            result = result && OperationBinary16iAutoTest(ARGS_OB16I(W + O, H - O, type, f1, f2));
            result = result && OperationBinary16iAutoTest(ARGS_OB16I(W - O, H + O, type, f1, f2));
        }

        return result;
    }

    bool OperationBinary16iAutoTest()
    {
        bool result = true;

        result = result && OperationBinary16iAutoTest(FUNC_OB16I(Simd::Base::OperationBinary16i), FUNC_OB16I(SimdOperationBinary16i));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && OperationBinary16iAutoTest(FUNC_OB16I(Simd::Sse2::OperationBinary16i), FUNC_OB16I(SimdOperationBinary16i));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && OperationBinary16iAutoTest(FUNC_OB16I(Simd::Avx2::OperationBinary16i), FUNC_OB16I(SimdOperationBinary16i));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && OperationBinary16iAutoTest(FUNC_OB16I(Simd::Vsx::OperationBinary16i), FUNC_OB16I(SimdOperationBinary16i));
#endif 

        return result;
    }

    bool VectorProductAutoTest(int width, int height, const FuncVP & f1, const FuncVP & f2)
    {
        bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

        View v(height, 1, View::Gray8, NULL, TEST_ALIGN(height));
        FillRandom(v);
        View h(width, 1, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(h);

        View d1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View d2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(v, h, d1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(v, h, d2));

        result = result && Compare(d1, d2, 0, true, 64);

        return result;
    }

    bool VectorProductAutoTest(const FuncVP & f1, const FuncVP & f2)
    {
        bool result = true;

        result = result && VectorProductAutoTest(W, H, f1, f2);
        result = result && VectorProductAutoTest(W - O, H + O, f1, f2);
        result = result && VectorProductAutoTest(W + O, H - O, f1, f2);

        return result;
    }

    bool VectorProductAutoTest()
    {
        bool result = true;

        result = result && VectorProductAutoTest(ARGS_VP(Simd::Base::VectorProduct), ARGS_VP(SimdVectorProduct));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && VectorProductAutoTest(ARGS_VP(Simd::Sse2::VectorProduct), ARGS_VP(SimdVectorProduct));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && VectorProductAutoTest(ARGS_VP(Simd::Avx2::VectorProduct), ARGS_VP(SimdVectorProduct));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && VectorProductAutoTest(ARGS_VP(Simd::Vsx::VectorProduct), ARGS_VP(SimdVectorProduct));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool OperationBinary8uDataTest(bool create, int width, int height, View::Format format, SimdOperationBinary8uType type, const FuncOB8U & f)
    {
        bool result = true;

        Data data(f.description);

        std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        View a(width, height, format, NULL, TEST_ALIGN(width));
        View b(width, height, format, NULL, TEST_ALIGN(width));
        View d1(width, height, format, NULL, TEST_ALIGN(width));
        View d2(width, height, format, NULL, TEST_ALIGN(width));

        if(create)
        {
            FillRandom(a);
            FillRandom(b);
            TEST_SAVE(a);
            TEST_SAVE(b);

            f.Call(a, b, d1, type);

            TEST_SAVE(d1);
        }
        else
        {
            TEST_LOAD(a);
            TEST_LOAD(b);
            TEST_LOAD(d1);

            f.Call(a, b, d2, type);

            TEST_SAVE(d2);

            result = result && Compare(d1, d2, 0, true, 64);
        }

        return result;
    }

    bool OperationBinary8uDataTest(bool create)
    {
        bool result = true;

        FuncOB8U f = FUNC_OB8U(SimdOperationBinary8u);
        for(SimdOperationBinary8uType type = SimdOperationBinary8uAverage; type <= SimdOperationBinary8uSaturatedAddition && result; type = SimdOperationBinary8uType(type + 1))
        {
            for(View::Format format = View::Gray8; format <= View::Bgra32; format = View::Format(format + 1))
            {
                std::string description = f.description + Data::Description(type) + Data::Description(format);
                result = result && OperationBinary8uDataTest(create, DW, DH, format, type, FuncOB8U(f.func, description));
            }
        }

        return result;
    }

    bool OperationBinary16iDataTest(bool create, int width, int height, SimdOperationBinary16iType type, const FuncOB16I & f)
    {
        bool result = true;

        Data data(f.description);

        std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        View a(width, height, View::Int16, NULL, TEST_ALIGN(width));
        View b(width, height, View::Int16, NULL, TEST_ALIGN(width));
        View d1(width, height, View::Int16, NULL, TEST_ALIGN(width));
        View d2(width, height, View::Int16, NULL, TEST_ALIGN(width));

        if(create)
        {
            FillRandom(a);
            FillRandom(b);
            TEST_SAVE(a);
            TEST_SAVE(b);

            f.Call(a, b, d1, type);

            TEST_SAVE(d1);
        }
        else
        {
            TEST_LOAD(a);
            TEST_LOAD(b);
            TEST_LOAD(d1);

            f.Call(a, b, d2, type);

            TEST_SAVE(d2);

            result = result && Compare(d1, d2, 0, true, 64);
        }

        return result;
    }

    bool OperationBinary16iDataTest(bool create)
    {
        bool result = true;

        FuncOB16I f = FUNC_OB16I(SimdOperationBinary16i);
        for(SimdOperationBinary16iType type = SimdOperationBinary16iAddition; type <= SimdOperationBinary16iAddition && result; type = SimdOperationBinary16iType(type + 1))
        {
            std::string description = f.description + Data::Description(type);
            result = result && OperationBinary16iDataTest(create, DW, DH, type, FuncOB16I(f.func, description));
        }

        return result;
    }

    bool VectorProductDataTest(bool create, int width, int height, const FuncVP & f)
    {
        bool result = true;

        Data data(f.description);

        std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        View v(height, 1, View::Gray8, NULL, TEST_ALIGN(height));
        View h(width, 1, View::Gray8, NULL, TEST_ALIGN(width));

        View d1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View d2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        if(create)
        {
            FillRandom(v);
            FillRandom(h);

            TEST_SAVE(v);
            TEST_SAVE(h);

            f.Call(v, h, d1);

            TEST_SAVE(d1);
        }
        else
        {
            TEST_LOAD(v);
            TEST_LOAD(h);

            TEST_LOAD(d1);

            f.Call(v, h, d2);

            TEST_SAVE(d2);

            result = result && Compare(d1, d2, 0, true, 64);
        }

        return result;
    }
}
