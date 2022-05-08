module;

#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include "glfw3.h"

export module rz.ui.rendering.glfw;

export namespace RayZath::UI::Render
{
	class Vulkan;

	class GLFW
	{
	private:
		Vulkan& m_vulkan;

		GLFWwindow* mp_window = nullptr;
		uint32_t m_extensions_count = 0;
		const char** m_extensions = nullptr;

	public:
		GLFW(Vulkan&);
		~GLFW();

	public:
		auto window() { return mp_window; }

		const char** extensions() { return m_extensions; }
		auto extensionsCount() { return m_extensions_count; }

		void init();

	};
}
