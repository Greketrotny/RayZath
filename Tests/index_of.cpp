#include "pch.h"
#include "CppUnitTest.h"

#include "../RayZath/index_of.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace Tests
{
	using index_of = RayZath::Utils::index_of;

	TEST_CLASS(StaticIndexOf)
	{
	public:
		TEST_METHOD(IndexOfValue)
		{
			Assert::AreEqual(index_of::value<10>::in_sequence<10>, std::size_t(0));
			Assert::AreEqual(index_of::value<10>::in_sequence<10, 20>, std::size_t(0));
			Assert::AreEqual(index_of::value<20>::in_sequence<10, 20>, std::size_t(1));
			Assert::AreEqual(index_of::value<30>::in_sequence<30, 20, 10>, std::size_t(0));
			Assert::AreEqual(index_of::value<20>::in_sequence<30, 20, 10>, std::size_t(1));
			Assert::AreEqual(index_of::value<10>::in_sequence<30, 20, 10>, std::size_t(2));
			Assert::AreEqual(index_of::value<20>::in_sequence<30, 20, 20>, std::size_t(1));
		}
		TEST_METHOD(IndexOfType)
		{
			Assert::AreEqual(index_of::type<int>::in_sequence<int>, std::size_t(0));
			Assert::AreEqual(index_of::type<int>::in_sequence<int, float>, std::size_t(0));
			Assert::AreEqual(index_of::type<float>::in_sequence<int, float>, std::size_t(1));
			Assert::AreEqual(index_of::type<int>::in_sequence<int, float, double>, std::size_t(0));
			Assert::AreEqual(index_of::type<float>::in_sequence<int, float, double>, std::size_t(1));
			Assert::AreEqual(index_of::type<double>::in_sequence<int, float, double>, std::size_t(2));
			Assert::AreEqual(index_of::type<float>::in_sequence<int, float, float>, std::size_t(1));
		}
	};
}
