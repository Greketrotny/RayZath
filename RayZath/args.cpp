#include "args.hpp"

#include <ranges>
#include <iostream>
#include <format>

namespace RayZath
{
	Args::Option::Option(std::string name, bool required, bool multiple)
		: name(std::move(name))
		, required(required)
		, multiple(multiple)
	{}

	Args::Option::operator std::string() const
	{
		std::stringstream ss;
		if (!required)
			ss << '[';
		ss << name;
		if (multiple)
			ss << "...";
		if (!required)
			ss << ']';
		return ss.str();
	}


	Args::Arg::Arg(std::set<std::string> variants, std::string description, std::vector<Option> options)
		: m_variants(std::move(variants))
		, m_description(std::move(description))
		, m_options(std::move(options))
	{
		if (m_variants.empty())
			throw std::invalid_argument("argument had 0 variants");
	}

	Args::Arg::operator std::string() const
	{
		std::stringstream ss;
		ss << *m_variants.begin();
		for (const auto& variant : m_variants | std::views::drop(1))
			ss << ", " << variant;

		for (const auto& option : m_options)
			ss << ' ' << std::string(option);
		return ss.str();
	}

	bool Args::Arg::hasVariant(const std::string& variant) const
	{
		return m_variants.contains(variant);
	}
	bool Args::Arg::hasRequiredOption() const
	{
		return std::ranges::find_if(m_options, [](const auto& arg) { return arg.required; }) != m_options.end();
	}


	Args& Args::arg(Args::Arg&& arg)
	{
		m_args.push_back(std::move(arg));
		return *this;
	}

	std::string Args::usageString()
	{
		std::vector<std::string> arg_strs;
		for (const auto& arg : m_args)
			arg_strs.push_back(std::string(arg));

		std::size_t max_width = 0;
		for (const auto& str : arg_strs)
			max_width = std::max(max_width, str.length());

		const std::string format = "  {:" + std::to_string(max_width) + "} {}\n";
		std::stringstream ss;
		ss << "Arguments:\n";
		for (std::size_t row = 0; row < m_args.size(); row++)
		{
			const auto& arg_str = arg_strs[row];
			const auto& desc_str = m_args[row].m_description;
			ss << std::vformat(std::string_view(format), std::make_format_args(arg_str, desc_str));
		}
		return ss.str();
	}

	Args::args_map_t Args::parse(const int argc, char* argv[])
	{
		return parse(argc, const_cast<const char**>(argv));
	}
	Args::args_map_t Args::parse(const std::size_t argc, const char* argv[])
	{
		std::vector<std::string_view> vec(argv, std::next(argv, argc));
		return parse(str_args_t(vec));
	}
	Args::args_map_t Args::parse(const str_args_t& arg_strs)
	{
		if (arg_strs.empty()) return {};

		using arg_map_t = std::map<std::string, std::vector<std::string>>;
		using arg_map_iterator_t = arg_map_t::iterator;
		arg_map_t arg_opt_map;
		arg_map_iterator_t arg_opt_map_iterator;

		auto arg_str_iterator = arg_strs.begin();

		do
		{
			auto arg_iterator = findArgument(*arg_str_iterator);
			if (arg_iterator == m_args.end())
				throw std::runtime_error("Unknown argument \"" + std::string(*arg_str_iterator) + "\".");

			bool inserted = false;
			std::tie(arg_opt_map_iterator, inserted) =
				arg_opt_map.insert({std::string(*arg_str_iterator), {}});
			if (!inserted)
				throw std::runtime_error("\"" + arg_opt_map_iterator->first + "\" argument passed more than once.");
			++arg_str_iterator;

			[&]() {
				for (const auto& option : arg_iterator->m_options)
				{
					if (option.required)
					{
						if (arg_str_iterator == arg_strs.end())
						{
							throw std::runtime_error(
								"Option \"" + option.name +
								"\" required for argument \"" + *arg_iterator->m_variants.begin() + "\".");
						}

						do
						{
							arg_opt_map_iterator->second.push_back(std::string(*arg_str_iterator++));
						} while (option.multiple && arg_str_iterator != arg_strs.end());
					}
					else
					{
						if (arg_str_iterator == arg_strs.end())
							return;

						do
						{
							auto potential_arg = findArgument(*arg_str_iterator);
							if (potential_arg != m_args.end())
								return;

							arg_opt_map_iterator->second.push_back(std::string(*arg_str_iterator++));

						} while (option.multiple && arg_str_iterator != arg_strs.end());
					}
				}
			}();
			
		} while (arg_str_iterator != arg_strs.end());

		return arg_opt_map;
	}
}
