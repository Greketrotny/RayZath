#define _CRTDBG_MAP_ALLOC  
#include <stdlib.h> 
#include <crtdbg.h>

#include <io.h>
#include <fcntl.h>

#include <iostream>
#include <stdexcept>
#include <span>

#include "args.hpp"

#include "rzexception.hpp"

#include "application.hpp"

int main(int argc, char* argv[])
{
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);

	try
	{
		auto arg_def = RayZath::Args{}
			.arg(RayZath::Args::Arg({"-h", "--help"}, "Prints help message.", {}))
			.arg(
				RayZath::Args::Arg(
					{"--headless"},
					"Renders given scene without UI.",
					{
						RayZath::Args::Option("scene_path", true),
						RayZath::Args::Option("report_path", false),
						RayZath::Args::Option("config_path", false)
					}));

		std::cout << arg_def.usageString();
		auto args = arg_def.parse(argc - 1, argv + 1);

		
		for (const auto& [arg, options] : args)
		{
			std::cout << arg << ": ";
			for (const auto& opt : options)
			{
				std::cout << opt << ", ";
			}
			std::cout << std::endl;
		}
	}
	catch (std::exception& ex)
	{
		std::cerr << ex.what() << '\n';
		return 1;
	}
	catch (...)
	{
		std::cout << "unknown error\n";
		return 1;
	}
}

int runHeadless(const std::filesystem::path& scene_path, const std::filesystem::path& report_path)
{
	std::cout << "headless rendering...";
	return 0;
}

int handleArgs()
{
	/*for (size_t arg_idx = 1; arg_idx < args.size(); arg_idx++)
	{
		const auto& arg = args[arg_idx];
		if (arg == "-h" || arg == "--help")
		{
			std::cout << std::format("{}", "options:\n");
			std::cout << std::format("  {:35} {}", "-h, --help", "Print this message.\n");
			std::cout << std::format(
				"  {:35} {}",
				"--headless scene_path [report path]", "Headless rendering of passed scenes.\n");
			return 0;
		}

		if (arg == "--headless")
		{
			if (arg_idx + 1 >= arg.size())
				throw std::runtime_error("[scene path] [report path] arguments expected for"s + arg.data());
			scene_path = args[arg_idx + 1];

			if (arg_idx + 2 < arg.size())
				report_path = args[arg_idx + 2];
			else
				report_path = std::filesystem::current_path();

			headless = true;
		}
	}

	if (headless)
	{
		return runHeadless(scene_path, report_path);
	}
	else
	{
		return RayZath::UI::Application::instance().run();
	}*/
	return 0;
}