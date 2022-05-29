module;

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include "glfw3.h"
#include "vulkan/vulkan.h"

#include <functional>

export module rz.ui.rendering;

export import rz.ui.rendering.glfw;
export import rz.ui.rendering.vulkan;

namespace RayZath::UI::Rendering
{
	export class RenderingWrapper
	{
	private:
	public:
		GLFW::GLFWWrapper m_glfw;
		Vulkan::VulkanWrapper m_vulkan;

		VkPipelineCache m_vk_pipeline_cache = VK_NULL_HANDLE;

		static constexpr int m_min_image_count = 2;

	public:
		RenderingWrapper();
		~RenderingWrapper();

		int run(std::function<void()> drawUi);
	};
}
