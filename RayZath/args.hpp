#ifndef RZ_ARGS_HPP
#define RZ_ARGS_HPP

#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <span>
#include <set>

namespace RayZath
{
	class Args
	{
	public:
		using str_args_t = std::span<const std::string_view>;
		using args_map_t = std::map<std::string, std::vector<std::string>>;

		struct Option
		{
			std::string name;
			bool required = true, multiple = false;

			Option(std::string name, bool required = true, bool multiple = false);

			explicit operator std::string() const;
		};
		struct Arg
		{
			std::set<std::string> m_variants;
			std::string m_description;
			std::vector<Option> m_options;

			Arg(std::set<std::string> variants, std::string description, std::vector<Option> options);

			explicit operator std::string() const;

			bool hasVariant(const std::string& variant) const;
			bool hasRequiredOption() const;
		};

	private:
		std::vector<Arg> m_args;

	public:
		Args& arg(Arg&& arg);
		args_map_t parse(const int argc, char* argv[]);
		args_map_t parse(const size_t argc, const char* argv[]);
		args_map_t parse(const str_args_t& args);
		
		std::string usageString();

	private:
		decltype(m_args)::iterator findArgument(const std::string_view& arg_str)
		{
			return std::ranges::find_if(m_args, [&](const auto& arg) {
				return arg.hasVariant(std::string(arg_str));
				});
		}
	};
}

#endif 
