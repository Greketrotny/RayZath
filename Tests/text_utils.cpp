#include "pch.h"
#include "CppUnitTest.h"

#include "../RayZath/text_utils.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace Tests
{
	TEST_CLASS(TextUtils)
	{
	public:
		TEST_METHOD(IntegerToScientific)
		{
			using namespace RayZath::Utils;
			using namespace std::string_literals;
			Assert::AreEqual("0.000"s, scientificWithPrefix(0));
			Assert::AreEqual("1.000"s, scientificWithPrefix(1));
			Assert::AreEqual("9.000"s, scientificWithPrefix(9));
			Assert::AreEqual("10.00"s, scientificWithPrefix(10));
			Assert::AreEqual("11.00"s, scientificWithPrefix(11));
			Assert::AreEqual("54.00"s, scientificWithPrefix(54));
			Assert::AreEqual("99.00"s, scientificWithPrefix(99));
			Assert::AreEqual("100.0"s, scientificWithPrefix(100));
			Assert::AreEqual("101.0"s, scientificWithPrefix(101));
			Assert::AreEqual("102.0"s, scientificWithPrefix(102));
			Assert::AreEqual("999.0"s, scientificWithPrefix(999));
			Assert::AreEqual("1.000K"s, scientificWithPrefix(1000));
			Assert::AreEqual("1.001K"s, scientificWithPrefix(1001));
			Assert::AreEqual("1.010K"s, scientificWithPrefix(1010));
			Assert::AreEqual("1.100K"s, scientificWithPrefix(1100));
			Assert::AreEqual("9.999K"s, scientificWithPrefix(9999));
			Assert::AreEqual("10.00K"s, scientificWithPrefix(10000));
			Assert::AreEqual("100.0K"s, scientificWithPrefix(100000));
			Assert::AreEqual("1.000M"s, scientificWithPrefix(1000000));
			Assert::AreEqual("10.00M"s, scientificWithPrefix(10000000));
			Assert::AreEqual("100.0M"s, scientificWithPrefix(100000000));
			Assert::AreEqual("1.000G"s, scientificWithPrefix(1000000000));
			Assert::AreEqual("10.00G"s, scientificWithPrefix(10000000000));
			Assert::AreEqual("100.0G"s, scientificWithPrefix(100000000000));
			Assert::AreEqual("1.000T"s, scientificWithPrefix(1000000000000));
			Assert::AreEqual("10.00T"s, scientificWithPrefix(10000000000000));
			Assert::AreEqual("100.0T"s, scientificWithPrefix(100000000000000));
			Assert::AreEqual("1.000P"s, scientificWithPrefix(1000000000000000));
			Assert::AreEqual("10.00P"s, scientificWithPrefix(10000000000000000));
			Assert::AreEqual("100.0P"s, scientificWithPrefix(100000000000000000));
			Assert::AreEqual("1.000E"s, scientificWithPrefix(1000000000000000000));
			Assert::AreEqual("10.00E"s, scientificWithPrefix(10000000000000000000));
			Assert::AreEqual("18.44E"s, scientificWithPrefix(18446744073709551615));
		}
	};
}
