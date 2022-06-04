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

export namespace RayZath::UI::Rendering::Vulkan
{
	class VulkanWrapper
	{
	private:
		 Vulkan::Instance m_instance;
		 GLFW::GLFWWrapper& m_glfw;	
	public:
		 Vulkan::Window m_window;

	public:
		bool m_swapchain_rebuild = false;
					
		//ImGui_ImplVulkanH_Window m_imgui_main_window;

		// --------------------
		uint32_t image_width, image_height;
		VkImage m_image = VK_NULL_HANDLE;
		VkImageView m_image_view = VK_NULL_HANDLE;
		VkSampler m_sampler = VK_NULL_HANDLE;
		VkDescriptorSetLayout m_descriptor_set_layout = VK_NULL_HANDLE;
		VkDeviceMemory m_device_memory = VK_NULL_HANDLE;
		ImTextureID m_render_texture = nullptr;

		VkBuffer m_buffer = VK_NULL_HANDLE;
		VkDeviceMemory m_staging_memory = VK_NULL_HANDLE;

	public:
		VulkanWrapper(GLFW::GLFWWrapper& glfw);
		~VulkanWrapper();

		auto& instance() { return m_instance; }
		auto& glfw() { return m_glfw; }
		auto& window() { return m_window; }

		static VkResult check(VkResult result);

		void init();
		void destroy();
	public:
		void frameRender();
		void framePresent();

		/*void createImage(const Graphics::Bitmap& bitmap, VkCommandBuffer command_buffer);
		void updateImage(const Graphics::Bitmap& bitmap, VkCommandBuffer command_buffer);
		void destroyImage();

	private:
		void createBuffer(
			VkDeviceSize size,
			VkBufferUsageFlags usage_flags,
			VkMemoryPropertyFlags properties,
			VkBuffer& buffer, VkDeviceMemory& buffer_memory);
		uint32_t findMemoryType(const uint32_t type_filter, const VkMemoryPropertyFlags properties);*/
	};
}
