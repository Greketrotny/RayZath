#pragma once

#include "instance.hpp"

#include "window.hpp"
#include "image.hpp"

namespace RayZath::UI::Rendering::Vulkan
{
	class VulkanWrapper
	{
	private:
		Vulkan::Instance m_instance;
		GLFW::GLFWWrapper& m_glfw;
	public:
		Vulkan::Window m_window;
		Image m_render_image;

	public:
		VulkanWrapper(GLFW::GLFWWrapper& glfw);

		auto& instance() { return m_instance; }
		auto& glfw() { return m_glfw; }
		auto& window() { return m_window; }

		static VkResult check(VkResult result);

		void init();
	public:
		void frameRender();
	};
}
