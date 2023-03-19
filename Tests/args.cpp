#include "pch.h"
#include "CppUnitTest.h"

#include "../RayZath/args.hpp"

#include <array>
#include <functional>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace RayZath;
using namespace std::string_literals;

namespace Tests
{
	TEST_CLASS(ArgumentParser)
	{
	public:
		TEST_METHOD(PassingArgumentWhenTakingNoneThrows)
		{
			Args args{};
			std::array arg_strs = {
				"-h"
			};
			Assert::ExpectException<std::runtime_error>([&]() {
				args.parse(arg_strs.size(), arg_strs.data());
				}, L"Unknown argument \"-h\".");
		}
		TEST_METHOD(PassingNothingWhenTakingNothingPasses)
		{
			Args args{};
			std::array<const char*, 0> arg_strs = {};
			auto map = args.parse(arg_strs.size(), arg_strs.data());
			Assert::IsTrue(map.empty());
		}

		TEST_METHOD(ThrowWhenRequiredOptionNotGiven)
		{
			auto args = Args{}.arg(Args::Arg({"-h"}, {"help"}, {Args::Option("option", true, false)}));
			std::array arg_strs = {
				"-h"
			};
			Assert::ExpectException<std::runtime_error>([&]() {
				args.parse(arg_strs.size(), arg_strs.data());
				}, L"Option \"option\" required for argument \"-h\".");
		}
		TEST_METHOD(ParseArgumentWithRequiredOption)
		{
			auto args = Args{}.arg(Args::Arg({"-h"}, {"help"}, {Args::Option("option", true, false)}));
			std::array arg_strs = {"-h", "help_option"};
			auto map{args.parse(arg_strs.size(), arg_strs.data())};
			Assert::IsTrue(map.contains("-h"));
			const auto& options = map["-h"];
			Assert::AreEqual(options.size(), std::size_t(1));
			Assert::AreEqual(options[0], "help_option"s);
		}
		TEST_METHOD(ParseArgumentWithTwoRequiredOptions)
		{
			auto args = Args{}
				.arg(Args::Arg({"-h"}, {"help"}, {
					Args::Option("option1", true, false),
					Args::Option("option2", true, false)}));
			std::array arg_strs = {"-h", "help_option", "option2"};
			auto map{args.parse(arg_strs.size(), arg_strs.data())};
			Assert::IsTrue(map.contains("-h"));
			const auto& options = map["-h"];
			Assert::AreEqual(options.size(), std::size_t(2));
			Assert::AreEqual(options[0], "help_option"s);
			Assert::AreEqual(options[1], "option2"s);
		}

		TEST_METHOD(ParseOptionalOptionsUpToANewArgument)
		{
			auto args = Args{}
				.arg(Args::Arg({"-h"}, {"help"}, {
					Args::Option("option1", true, false),
					Args::Option("option2", false, false)}))
					.arg(Args::Arg({"-a"}, {"arg"}, {}));
			std::array arg_strs = {"-h", "option1", "-a"};
			auto map{args.parse(arg_strs.size(), arg_strs.data())};
			Assert::IsTrue(map.contains("-h"));
			const auto& h_options = map["-h"];
			Assert::AreEqual(h_options.size(), std::size_t(1));
			Assert::AreEqual(h_options[0], "option1"s);
			Assert::IsTrue(map.contains("-a"));
			const auto& a_options = map["-a"];
			Assert::AreEqual(a_options.size(), std::size_t(0));
		}
		TEST_METHOD(GreedilyParseRequiredMultiOption)
		{
			auto args = Args{}
				.arg(
					Args::Arg({"-h"}, {"help"}, {
					Args::Option("option1", true, true)
						}))
				.arg(Args::Arg({"-a"}, {"arg"}, {}));
			std::array arg_strs = {"-h", "option1", "-a"};
			auto map{args.parse(arg_strs.size(), arg_strs.data())};
			Assert::IsTrue(map.contains("-h"));
			const auto& h_options = map["-h"];
			Assert::AreEqual(h_options.size(), std::size_t(2));
			Assert::AreEqual(h_options[0], "option1"s);
			Assert::AreEqual(h_options[1], "-a"s);
		}
	};
}
