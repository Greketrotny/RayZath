module;

#include "vulkan/vulkan.h"
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include "glfw3.h"

#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "vec2.h"
#include "bitmap.h"

#include <iostream>
#include <string>
#include <limits>
#include <format>
#include <unordered_map>
#include <memory>

export module rz.ui.rendering.vulkan;

import rz.ui.rendering.glfw;
import rz.ui.rendering.vulkan.instance;
import rz.ui.rendering.vulkan.window;
import rz.ui.rendering.vulkan.image;

export namespace RayZath::UI::Rendering::Vulkan
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
