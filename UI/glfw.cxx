module;

#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include "glfw3.h"

#include "rzexception.h"
#include "vec2.h"

#include <iostream>
#include <string>
#include <limits>
#include <format>
#include <unordered_map>
#include <memory>

module rz.ui.rendering.backend;

namespace RayZath::UI::Render
{
	void errorCallback(const int error, const char* const description)
	{
		std::cout << "[glfw] error: " << error << ": " << description;
	}

	GLFW::GLFW(Vulkan& vulkan)
		: m_vulkan(vulkan)
	{}
	GLFW::~GLFW()
	{
		glfwDestroyWindow(window());
		glfwTerminate();
	}

	Math::vec2ui32 GLFW::frameBufferSize()
	{
		int width, height;
		glfwGetFramebufferSize(window(), &width, &height);
		return Math::vec2ui32(width, height);
	}
	Math::vec2ui32 GLFW::windowSize()
	{
		int width, height;
		glfwGetWindowSize(window(), &width, &height);
		return Math::vec2ui32(width, height);
	}

	void GLFW::init()
	{
		glfwSetErrorCallback(errorCallback);
		RZAssert(glfwInit(), "failed to initialize glfw");

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		mp_window = glfwCreateWindow(1280, 720, "Dear ImGui GLFW+Vulkan example", NULL, NULL);

		RZAssert(glfwVulkanSupported(), "vulkan in glfw is not supported");

		m_extensions = glfwGetRequiredInstanceExtensions(&m_extensions_count);
	}

	VkSurfaceKHR GLFW::createWindowSurface()
	{
		VkSurfaceKHR surface;
		Vulkan::check(glfwCreateWindowSurface(m_vulkan.m_instance, window(), m_vulkan.mp_allocator, &surface));
		return surface;
	}
}
