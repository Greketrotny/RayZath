#pragma once

#include "instance.hpp"

#include "window.hpp"
#include "image.hpp"

namespace RayZath::UI::Rendering::Vulkan
{
	class Module
	{
	private:
		GLFW::Module& m_glfw;
	public:
		Vulkan::Window m_window;
		Image m_render_image;

	public:
		Module(GLFW::Module& glfw);

		auto& glfw() { return m_glfw; }
		auto& window() { return m_window; }

		static VkResult check(VkResult result);

		void init();
	public:
		void frameRender();
	};
}
