module;

#include "vulkan/vulkan.h"
#include "imgui_impl_vulkan.h"

#include "vec2.h"
#include "rzexception.h"
#include "bitmap.h"

#include <iostream>
#include <string>
#include <limits>
#include <format>
#include <unordered_map>
#include <memory>
#include <array>

module rz.ui.rendering.vulkan;

namespace RayZath::UI::Rendering::Vulkan
{
	VkResult VulkanWrapper::check(VkResult result)
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

	VulkanWrapper::VulkanWrapper(GLFW::GLFWWrapper& glfw)
		: m_glfw(glfw)
		, m_window(m_instance)
		, m_render_image(m_instance)
	{}

	void VulkanWrapper::init()
	{
		m_instance.init(m_glfw);
		m_window.init(m_glfw, m_glfw.frameBufferSize());
	}

	void VulkanWrapper::frameRender()
	{
		m_window.beginRenderPass();

		ImGui_ImplVulkan_RenderDrawData(
			ImGui::GetDrawData(),
			m_window.currentFrame().commandBuffer());

		m_window.endRenderPass();
	}
}
