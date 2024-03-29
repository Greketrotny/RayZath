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
#include "headless.hpp"

int main(int argc, char* argv[])
{
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);

	try
	{
		int run(int, char* []);
		return run(argc, argv);
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

int run(const int argc, char* argv[])
{
	auto arg_def = RayZath::Args{}
		.arg(RayZath::Args::Arg({"-h", "--help"}, "Prints help message.", {}))
		.arg(
			RayZath::Args::Arg(
				{"--headless"},
				"Execute rendering tasks without UI and generate a report.",
				{
					RayZath::Args::Option("task_path", true),
					RayZath::Args::Option("report_path", false)
				}))
		.arg(
			RayZath::Args::Arg(
				{"-r", "--render"},
				"When specified --headless, also saves rendered images.", {}));
	auto args{arg_def.parse(argc - 1, argv + 1)};

	if (args.contains("-h") || args.contains("--help"))
	{
		std::cout << arg_def.usageString() << std::endl;
		return 0;
	}

	if (args.contains("--headless"))
	{
		const auto& headless_params = args["--headless"];
		std::filesystem::path scene_path{}, report_path{};
		if (headless_params.size() > 0) scene_path.assign(headless_params[0]);
		if (headless_params.size() > 1) report_path.assign(headless_params[1]);
		return RayZath::Headless::Headless::instance().run(scene_path, report_path, args.contains("-r"));
	}
	else
	{
		return RayZath::UI::Application::instance().run();
	}	
}
