#include "pch.h"
#include "CppUnitTest.h"

#include "../RayZath/index_of.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace Tests
{
	using index_of = RayZath::Utils::index_of;

	TEST_CLASS(StaticIndexOf)
	{
	public:
		TEST_METHOD(IndexOfValue)
		{
			Assert::AreEqual(index_of::value<10>::in_sequence<10>, size_t(0));
			Assert::AreEqual(index_of::value<10>::in_sequence<10, 20>, size_t(0));
			Assert::AreEqual(index_of::value<20>::in_sequence<10, 20>, size_t(1));
			Assert::AreEqual(index_of::value<30>::in_sequence<30, 20, 10>, size_t(0));
			Assert::AreEqual(index_of::value<20>::in_sequence<30, 20, 10>, size_t(1));
			Assert::AreEqual(index_of::value<10>::in_sequence<30, 20, 10>, size_t(2));
			Assert::AreEqual(index_of::value<20>::in_sequence<30, 20, 20>, size_t(1));
		}
		TEST_METHOD(IndexOfType)
		{
			Assert::AreEqual(index_of::type<int>::in_sequence<int>, size_t(0));
			Assert::AreEqual(index_of::type<int>::in_sequence<int, float>, size_t(0));
			Assert::AreEqual(index_of::type<float>::in_sequence<int, float>, size_t(1));
			Assert::AreEqual(index_of::type<int>::in_sequence<int, float, double>, size_t(0));
			Assert::AreEqual(index_of::type<float>::in_sequence<int, float, double>, size_t(1));
			Assert::AreEqual(index_of::type<double>::in_sequence<int, float, double>, size_t(2));
			Assert::AreEqual(index_of::type<float>::in_sequence<int, float, float>, size_t(1));
		}
	};
}
