#include "pch.h"
#include "CppUnitTest.h"

#include "../RayZath/dictionary.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace Tests
{
	using RayZath::Utils::static_dictionary;

	template <typename T>
	using tt_dictionary = static_dictionary::tt_translate<T>::template with<
		static_dictionary::tt_translation<int32_t, float>,
		static_dictionary::tt_translation<int64_t, double>>::value;
	template <typename T>
	constexpr auto tv_dictionary = static_dictionary::tv_translate<T>::template with<
		static_dictionary::tv_translation<float, 10>,
		static_dictionary::tv_translation<double, 20>>::value;
	template <auto V>
	using vt_dictionary = static_dictionary::vt_translate<V>::template with<
		static_dictionary::vt_translation<4, float>,
		static_dictionary::vt_translation<8, double>>::value;
	template <auto V>
	constexpr auto vv_dictionary = static_dictionary::vv_translate<V>::template with<
		static_dictionary::vv_translation<10, 1 << 10>,
		static_dictionary::vv_translation<20, 1 << 20>,
		static_dictionary::vv_translation<30, 1 << 30>>::value;


	constexpr const char float_name[] = "float";
	constexpr const char int_name[] = "int";
	template <typename T>
	constexpr auto name_dictionary = static_dictionary::tv_translate<T>::template with<
		static_dictionary::tv_translation<float, float_name>,
		static_dictionary::tv_translation<int, int_name>>::value;

	TEST_CLASS(StaticDictionary)
	{
	public:
		TEST_METHOD(TypeTypeTranslation)
		{
			Assert::IsTrue(std::is_same_v<tt_dictionary<int32_t>, float>);
			Assert::IsTrue(std::is_same_v<tt_dictionary<int64_t>, double>);
		}
		TEST_METHOD(TypeValueTranslation)
		{
			Assert::AreEqual(10, tv_dictionary<float>);
			Assert::AreEqual(20, tv_dictionary<double>);
		}
		TEST_METHOD(ValueTypeTranslation)
		{
			Assert::IsTrue(std::is_same_v<vt_dictionary<4>, float>);
			Assert::IsTrue(std::is_same_v<vt_dictionary<8>, double>);
		}
		TEST_METHOD(ValueValueTranslation)
		{
			Assert::AreEqual(1 << 10, vv_dictionary<10>);
			Assert::AreEqual(1 << 20, vv_dictionary<20>);
			Assert::AreEqual(1 << 30, vv_dictionary<30>);
		}
		TEST_METHOD(ValueValueTranslationWithCharArray)
		{
			Assert::AreEqual("float", name_dictionary<float>);
			Assert::AreEqual("int", name_dictionary<int>);
		}
	};
}
