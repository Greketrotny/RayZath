#ifndef STATIC_DICTIONARY_H
#define STATIC_DICTIONARY_H

#include <type_traits>

	struct static_dictionary
	{
	private:
		template <bool provide, typename tr>
		struct provide_type_if;
		template <typename tr>
		struct provide_type_if<false, tr> {};
		template <typename tr>
		struct provide_type_if<true, tr> { using value = typename tr::value; };

		template <bool provide, typename tr>
		struct provide_value_if;
		template <typename tr>
		struct provide_value_if<false, tr> {};
		template <typename tr>
		struct provide_value_if<true, tr> { static constexpr auto value = tr::value; };

	public:
		template <typename Key, typename Value>
		struct tt_translation
		{
			using key = Key;
			using value = Value;
		};
		template <typename Key, auto Value>
		struct tv_translation
		{
			using key = Key;
			static constexpr auto value = Value;
		};
		template <auto Key, typename Value>
		struct vt_translation
		{
			static constexpr auto key = Key;
			using value = Value;
		};
		template <auto Key, auto Value>
		struct vv_translation
		{
			static constexpr auto key = Key;
			static constexpr auto value = Value;
		};

		template <typename Key>
		struct tt_translate
		{
			template <typename... trs>
			struct with : public provide_type_if<std::is_same_v<Key, typename trs::key>, trs>...
			{
			private:
				static constexpr auto translation_count = (... + std::size_t(std::is_same_v<Key, typename trs::key>));
				static_assert(translation_count != 0, "no translation found for given key");
				static_assert(translation_count == 1, "more than one translation found for given key");
			};
		};
		template <typename Key>
		struct tv_translate
		{
			template <typename... trs>
			struct with : public provide_value_if<std::is_same_v<Key, typename trs::key>, trs>...
			{
			private:
				static constexpr auto translation_count = (... + std::size_t(std::is_same_v<Key, typename trs::key>));
				static_assert(translation_count != 0, "no translation found for given key");
				static_assert(translation_count == 1, "more than one translation found for given key");
			};
		};
		template <auto Key>
		struct vt_translate
		{
			template <typename... trs>
			struct with : public provide_type_if<Key == trs::key, trs>...
			{
			private:
				static constexpr auto translation_count = (... + std::size_t(Key == trs::key));
				static_assert(translation_count != 0, "no translation found for given key");
				static_assert(translation_count == 1, "more than one translation found for given key");
			};
		};
		template <auto Key>
		struct vv_translate
		{
			template <typename... trs>
			struct with : public provide_value_if<Key == trs::key, trs>...
			{
			private:
				static constexpr auto translation_count = (... + std::size_t(Key == trs::key));
				static_assert(translation_count != 0, "no translation found for given key");
				static_assert(translation_count == 1, "more than one translation found for given key");
			};
		};
	};

	struct is
	{
		template <auto V>
		struct value
		{
			template <auto... Vs>
			struct any_of
			{
				static constexpr bool value = (... || (V == Vs));
			};
			template <auto... Vs>
			struct all_of
			{
				static constexpr bool value = (... && (V == Vs));
			};
		};
		template <typename T>
		struct type
		{
			template <typename... Ts>
			struct any_of
			{
				static constexpr bool value = (... || (std::is_same_v<T, Ts>));
			};
			template <typename... Ts>
			struct all_of
			{
				static constexpr bool value = (... && (std::is_same_v<T, Ts>));
			};
		};
	};


#endif
