import rz.ui.application;

import <iostream>;
import <stdexcept>;

int main(int, char**)
{
    try
    {
        return RayZath::UI::Application::instance().run();
    }
    catch (std::exception& ex)
    {
        //std::cerr << ex.what() << '\n';
        return 1;
    }
}
