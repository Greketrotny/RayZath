#include "vulkan.hpp"

#include "imgui_impl_vulkan.h"

#include "rzexception.hpp"

#include "vec2.h"
#include "bitmap.h"

#include <iostream>
#include <string>
#include <format>

namespace RayZath::UI::Rendering::Vulkan
{
	VkResult Module::check(VkResult result)
	{
		if (result == VkResult::VK_SUCCESS) return result;

		using vkresult_int_t = std::underlying_type<VkResult>::type;
		if (vkresult_int_t(result) > 0)
		{
			// TODO: log positive result
			// return result;
		}
		RZThrow(std::format("Vulkan error {}", vkresult_int_t(result)));
	}

	Module::Module(GLFW::Module& glfw)
		: m_glfw(glfw)
		, m_vulkan_instance_initializer(m_glfw)
		, m_window(m_glfw, m_glfw.frameBufferSize())
	{}

	void Module::frameRender()
	{
		m_window.beginRenderPass();

		ImGui_ImplVulkan_RenderDrawData(
			ImGui::GetDrawData(),
			m_window.currentFrame().commandBuffer());

		m_window.endRenderPass();
	}
}
