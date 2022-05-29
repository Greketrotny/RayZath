module;

#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include "glfw3.h"

#include "rzexception.h"
#include "vec2.h"

#include <iostream>
#include <string>

module rz.ui.rendering.glfw;

namespace RayZath::UI::Rendering::GLFW
{
	void errorCallback(const int error, const char* const description)
	{
		std::cout << "[glfw] error: " << error << ": " << description;
	}

	GLFWWrapper::~GLFWWrapper()
	{
		glfwDestroyWindow(window());
		glfwTerminate();
	}

	Math::vec2ui32 GLFWWrapper::frameBufferSize()
	{
		int width, height;
		glfwGetFramebufferSize(window(), &width, &height);
		return Math::vec2ui32(width, height);
	}
	Math::vec2ui32 GLFWWrapper::windowSize()
	{
		int width, height;
		glfwGetWindowSize(window(), &width, &height);
		return Math::vec2ui32(width, height);
	}

	void GLFWWrapper::init()
	{
		glfwSetErrorCallback(errorCallback);
		RZAssert(glfwInit(), "failed to initialize glfw");

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		mp_window = glfwCreateWindow(1280, 720, "Dear ImGui GLFW+Vulkan example", NULL, NULL);

		RZAssert(glfwVulkanSupported(), "vulkan in glfw is not supported");

		m_extensions = glfwGetRequiredInstanceExtensions(&m_extensions_count);
	}

	VkSurfaceKHR GLFWWrapper::createWindowSurface(
		VkInstance instance,
		VkAllocationCallbacks* allocator)
	{
		VkSurfaceKHR surface;
		glfwCreateWindowSurface(instance, window(), allocator, &surface);
		return surface;
	}
}
