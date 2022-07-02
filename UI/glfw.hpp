#pragma once

#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include "glfw3.h"

#include "vec2.h"

namespace RayZath::UI::Rendering::GLFW
{
	class Module
	{
	private:
		GLFWwindow* mp_window = nullptr;
		uint32_t m_extensions_count = 0;
		const char** m_extensions = nullptr;

	public:
		~Module();

	public:
		GLFWwindow* window() { return mp_window; }

		const char** extensions() { return m_extensions; }
		uint32_t extensionsCount() { return m_extensions_count; }
		Math::vec2ui32 frameBufferSize();
		Math::vec2ui32 windowSize();

		void init();

		VkSurfaceKHR createWindowSurface(
			VkInstance instance,
			VkAllocationCallbacks* allocator);
	};
}
