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
		Vulkan::Instance::Initializer m_vulkan_instance_initializer;
	public:
		Vulkan::Window m_window;

	public:
		Module(GLFW::Module& glfw);

		auto& glfw() { return m_glfw; }
		auto& window() { return m_window; }

		static VkResult check(VkResult result);
	public:
		void frameRender();
	};
}
