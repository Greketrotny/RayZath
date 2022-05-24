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
		return m_rendering.run();
	}

	void Application::draw()
	{
		m_viewport.draw(
			m_rendering.m_vulkan.m_render_texture,
			m_rendering.m_vulkan.image_width, m_rendering.m_vulkan.image_height);
	}
}
