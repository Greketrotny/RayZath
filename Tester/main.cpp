#define _CRTDBG_MAP_ALLOC  
#include <stdlib.h> 
#include <crtdbg.h>

#include "application.h"

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInst, LPWSTR args, INT ncmd)
{
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);

    Tester::Application app;
    app.Start();

    return 0;
}