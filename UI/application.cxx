module;

#include "rayzath.h"

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
		return m_rendering.run([this]() {
			this->draw();
			});
	}

	void Application::draw()
	{
		m_viewport.draw(m_rendering.m_vulkan.m_render_image);
		m_rendering.m_vulkan.desired_resolution = m_viewport.m_resolution;
	}
}
