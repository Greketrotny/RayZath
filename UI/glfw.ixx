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

	public:
		GLFW(Vulkan&);

	public:
		auto window() { return mp_window; }

	};
}
