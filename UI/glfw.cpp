#include "glfw.hpp"

#include "rzexception.h"

#include <iostream>
#include <string>

namespace RayZath::UI::Rendering::GLFW
{
	void errorCallback(const int error, const char* const description)
	{
		std::cout << "[glfw] error: " << error << ": " << description;
	}

	Module::~Module()
	{
		glfwDestroyWindow(window());
		glfwTerminate();
	}

	Math::vec2ui32 Module::frameBufferSize()
	{
		int width, height;
		glfwGetFramebufferSize(window(), &width, &height);
		return Math::vec2ui32(width, height);
	}
	Math::vec2ui32 Module::windowSize()
	{
		int width, height;
		glfwGetWindowSize(window(), &width, &height);
		return Math::vec2ui32(width, height);
	}
	bool Module::iconified()
	{
		return bool(glfwGetWindowAttrib(mp_window, GLFW_ICONIFIED));
	}
	bool Module::maximized()
	{
		return bool(glfwGetWindowAttrib(mp_window, GLFW_MAXIMIZED));
	}


	void Module::init()
	{
		glfwSetErrorCallback(errorCallback);
		RZAssert(glfwInit(), "failed to initialize glfw");

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		mp_window = glfwCreateWindow(1280, 720, "RayZath", NULL, NULL);
		RZAssert(mp_window, "glfwWindow creation failed");
		RZAssert(glfwVulkanSupported(), "vulkan in glfw is not supported");

		glfwMaximizeWindow(mp_window);

		m_extensions = glfwGetRequiredInstanceExtensions(&m_extensions_count);
	}
	void Module::setTitle(const std::string& title)
	{
		glfwSetWindowTitle(mp_window, title.c_str());
	}

	VkSurfaceKHR Module::createWindowSurface(
		VkInstance instance,
		VkAllocationCallbacks* allocator)
	{
		VkSurfaceKHR surface;
		glfwCreateWindowSurface(instance, window(), allocator, &surface);
		return surface;
	}
}
