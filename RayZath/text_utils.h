#ifndef TEXT_UTILS_H
#define TEXT_UTILS_H

#include <string>
#include <array>
#include <charconv>

namespace RayZath::Utils
{
	inline std::string scientificWithPrefix(const size_t value)
	{
		std::array<char, 24> value_str{};
		std::fill_n(value_str.begin(), 5, '0');
		auto result = std::to_chars(value_str.data(), value_str.data() + value_str.size(), value);
		if (result.ec != std::errc{})
			return {};

		auto end = result.ptr;
		const auto digits = end - value_str.data();

		const char* prefixes = "####KKKMMMGGGTTTPPPEEE";
		const auto prefix = prefixes[digits];
		const auto decimal_idx = (digits - 1) % 3 + 1;

		for (int i = 3; i >= decimal_idx; i--)
			value_str[i + 1] = value_str[i];
		value_str[decimal_idx] = '.';

		if (value >= 1000)
		{
			value_str[5] = prefix;
			return std::string(value_str.data(), 6);
		}
		else
		{
			return std::string(value_str.data(), 5);
		}
	}
}

#endif
