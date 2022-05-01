module rz.ui.application;

namespace RayZath::UI
{
    Application& Application::instance()
    {
        static Application application{};
        return application;
    }

    int Application::run()
    {
        return m_rendering.run();
    }
}
