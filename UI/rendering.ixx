module;

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include "glfw3.h"
#include "vulkan/vulkan.h"

export module rz.ui.rendering;

import rz.ui.rendering.backend;

export namespace RayZath::UI
{
	class Rendering
	{
	private:
		Render::GLFW m_glfw;
		Render::Vulkan m_vulkan;

		VkPipelineCache m_vk_pipeline_cache = VK_NULL_HANDLE;

		ImGui_ImplVulkanH_Window m_imgui_main_window;

		static constexpr int m_min_image_count = 2;
		bool g_SwapChainRebuild = false;

	public:
		Rendering();
		~Rendering();

		int run();

	private:
		void SetupVulkanWindow(ImGui_ImplVulkanH_Window* wd, VkSurfaceKHR surface, int width, int height);
		void CleanupVulkan();
		void CleanupVulkanWindow();

		void FrameRender(ImGui_ImplVulkanH_Window* wd, ImDrawData* draw_data);
		void FramePresent(ImGui_ImplVulkanH_Window* wd);
	};
}
