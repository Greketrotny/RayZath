#define _CRTDBG_MAP_ALLOC  
#include <stdlib.h> 
#include <crtdbg.h>

#include <io.h>
#include <fcntl.h>

#include <iostream>
#include <stdexcept>

#include "rzexception.hpp"

#include "application.hpp"

int main(int, char**)
{
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);

	try
	{
		return RayZath::UI::Application::instance().run();
	}
	catch (RayZath::Exception& ex) 
	{
		std::cerr << ex.ToString() << '\n';
	}
	catch (std::exception& ex)
	{
		std::cerr << ex.what() << '\n';
		return 1;
	}
}
