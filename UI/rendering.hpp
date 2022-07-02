#pragma once

#include "glfw.hpp"
#include "vulkan.hpp"

#include <functional>

namespace RayZath::UI::Rendering
{
	class RenderingWrapper
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
