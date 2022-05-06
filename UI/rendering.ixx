module;

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include "glfw3.h"
#include "vulkan/vulkan.h"

export module rz.ui.rendering;

import <stdexcept>;
import <iostream>;
import <string>;

export namespace RayZath::UI
{
	class Rendering
	{
	private:
		VkAllocationCallbacks* mp_vk_allocator = nullptr;
		VkInstance					m_vk_instance = VK_NULL_HANDLE;
		VkPhysicalDevice			m_vk_physical_device = VK_NULL_HANDLE;
		VkDevice					m_vk_device = VK_NULL_HANDLE;
		uint32_t					m_queue_family = (uint32_t)-1;
		VkQueue						m_vk_quque = VK_NULL_HANDLE;
		VkDebugReportCallbackEXT	m_vk_debug_report = VK_NULL_HANDLE;
		VkPipelineCache				m_vk_pipeline_cache = VK_NULL_HANDLE;
		VkDescriptorPool			m_vk_descriptor_pool = VK_NULL_HANDLE;

		ImGui_ImplVulkanH_Window m_imgui_main_window;

		static constexpr int m_min_image_count = 2;
		bool g_SwapChainRebuild = false;

		GLFWwindow* mp_glfw_window = nullptr;

	public:
		Rendering();
		~Rendering();

		int run();

	private:
		void SetupVulkan(const char** extensions, uint32_t extensions_count);
		void SetupVulkanWindow(ImGui_ImplVulkanH_Window* wd, VkSurfaceKHR surface, int width, int height);
		void CleanupVulkan();
		void CleanupVulkanWindow();

		void FrameRender(ImGui_ImplVulkanH_Window* wd, ImDrawData* draw_data);
		void FramePresent(ImGui_ImplVulkanH_Window* wd);
	};
}
