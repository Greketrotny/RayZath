#pragma once

#include "glfw.hpp"
#include "vulkan.hpp"

#include <functional>

namespace RayZath::UI::Rendering
{
	class Module
	{
	private:
	public:
		GLFW::Module m_glfw;
		Vulkan::Module m_vulkan;

		VkPipelineCache m_vk_pipeline_cache = VK_NULL_HANDLE;

		static constexpr int m_min_image_count = 2;

	public:
		Module();
		~Module();
	
		int run(std::function<void()> update, std::function<void()> render);

		void setWindowTitle(const std::string& title);
	private:
		void setImguiStyle();
	};
}
