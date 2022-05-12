module;

#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include "glfw3.h"

#include <iostream>
#include <format>

#include "rzexception.h"

module rz.ui.rendering.glfw;

import rz.ui.rendering.vulkan;

namespace RayZath::UI::Render
{
	void errorCallback(const int error, const char* const description)
	{
		std::cerr << "[glfw] error: " << error << ": " << description;
	}

	GLFW::GLFW(Vulkan& vulkan)
		: m_vulkan(vulkan)
	{}
	GLFW::~GLFW()
	{
		glfwDestroyWindow(window());
		glfwTerminate();
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
