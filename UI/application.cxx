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

    void Application::draw(Graphics::Bitmap&& bitmap)
    {
        m_viewport.draw(std::move(bitmap));
    }
}
