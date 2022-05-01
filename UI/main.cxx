#include <iostream>
#include <stdexcept>

import rz.ui.application;

int main(int, char**)
{
    try
    {
        return RayZath::UI::Application::instance().run();
    }
    catch (std::exception& ex)
    {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
