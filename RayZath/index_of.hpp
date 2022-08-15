#ifndef INDEX_OF_H
#define INDEX_OF_H

#include <type_traits>

namespace RayZath::Utils
{
	struct index_of
	{
		template <auto ToFind>
		struct value
		{
		private:
			template <bool Last, bool Found, std::size_t I, auto V, auto V1, auto... Vs>
			struct index;
			template <bool Found, std::size_t I, auto V, auto First, auto... Vs>
			struct index<true, Found, I, V, First, Vs...>
			{
				static_assert(V == First || Found, "value not found in given sequence");
				static constexpr std::size_t value = I;
			};
			template <bool Found, std::size_t I, auto V, auto First, auto... Vs>
			struct index<false, Found, I, V, First, Vs...>
			{
				static constexpr std::size_t value = (V == First) ? 
					I : 
					index<sizeof...(Vs) == 1u, Found || (V == First), I + 1, V, Vs...>::value;
			};

		public:
			template <auto... Vs>
			static constexpr auto in_sequence = index<sizeof...(Vs) == 1u, false, 0, ToFind, Vs...>::value;
		};

		template <typename ToFind>
		struct type
		{
		private:
			template <bool Last, bool Found, std::size_t I, typename T, typename T1, typename... Ts>
			struct index;
			template <bool Found, std::size_t I, typename T, typename First, typename... Ts>
			struct index<true, Found, I, T, First, Ts...>
			{
				static_assert(std::is_same_v<T, First> || Found, "type not found in given sequence");
				static constexpr std::size_t value = I;
			};
			template <bool Found, std::size_t I, typename T, typename First, typename... Ts>
			struct index<false, Found, I, T, First, Ts...>
			{
			private:
				static constexpr bool same = std::is_same_v<T, First>;
			public:
				static constexpr std::size_t value = same ?
					I :
					index<sizeof...(Ts) == 1u, Found || same, I + 1, T, Ts...>::value;
			};

		public:
			template <typename... Ts>
			static constexpr auto in_sequence = index<sizeof...(Ts) == 1u, false, 0, ToFind, Ts...>::value;
		};
	};
}

#endif
