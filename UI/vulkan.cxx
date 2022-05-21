module;

#include "vulkan/vulkan.h"
#include "imgui_impl_vulkan.h"

#include "rzexception.h"

#include <iostream>
#include <string>
#include <limits>
#include <format>
#include <unordered_map>
#include <memory>
#include <array>

module rz.ui.rendering.backend;

namespace RayZath::UI::Render
{
	Vulkan::Vulkan(GLFW& glfw)
		: m_glfw(glfw)
	{}
	Vulkan::~Vulkan()
	{
		destroyDescriptorPool();
		destroyLogicalDevice();
		destroyInstance();
	}

	VkResult Vulkan::check(VkResult result)
	{
		if (result == VkResult::VK_SUCCESS) return result;

		using vkresult_int_t = std::underlying_type<VkResult>::type;
		if (vkresult_int_t(result) > 0)
		{
			// TODO: log positive result
			// return result;
		}
		RZThrow(std::format("Vulkan error {}", vkresult_int_t(result)));
	}

	void Vulkan::init()
	{
		createInstance();
		selectPhysicalDevice();
		createLogicalDevice();
		createDescriptorPool();
		createWindowSurface();
	}
	void Vulkan::createInstance()
	{
		RZAssert(m_instance == VK_NULL_HANDLE, "Vulkan instance is already created");

		const char** extensions = m_glfw.extensions();
		const uint32_t extensions_count = m_glfw.extensionsCount();

		if constexpr (VULKAN_DEBUG_REPORT)
		{
			// emable debug extension
			std::unique_ptr<const char* [], decltype(std::free)*> extensions_with_debug(
				(const char**)malloc(sizeof(const char*) * (extensions_count + 1)),
				std::free);
			std::memcpy(extensions_with_debug.get(), extensions, extensions_count * sizeof(const char*));
			extensions_with_debug[extensions_count] = "VK_EXT_debug_report";

			// enable validation layers
			const char* layers[] = { "VK_LAYER_KHRONOS_validation" };

			// create vulkan instance
			VkInstanceCreateInfo instance_ci{};
			instance_ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
			instance_ci.enabledExtensionCount = extensions_count + 1;
			instance_ci.ppEnabledExtensionNames = extensions_with_debug.get();
			instance_ci.enabledLayerCount = 1;
			instance_ci.ppEnabledLayerNames = layers;
			check(vkCreateInstance(&instance_ci, mp_allocator, &m_instance));

			// setup debug report callback
			auto vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(
				m_instance, "vkCreateDebugReportCallbackEXT");
			RZAssert(vkCreateDebugReportCallbackEXT != NULL, "Failed to get debug report callback");

			VkDebugReportCallbackCreateInfoEXT debug_report_ci{};
			debug_report_ci.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT;
			debug_report_ci.flags =
				VK_DEBUG_REPORT_ERROR_BIT_EXT |
				VK_DEBUG_REPORT_WARNING_BIT_EXT |
				VK_DEBUG_REPORT_DEBUG_BIT_EXT |
				VK_DEBUG_REPORT_INFORMATION_BIT_EXT |
				VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
			debug_report_ci.pfnCallback = Vulkan::debugReport;
			check(vkCreateDebugReportCallbackEXT(m_instance, &debug_report_ci, mp_allocator, &m_debug_report_callback));
		}
		else
		{
			// create vulkan instance
			VkInstanceCreateInfo instance_ci{};
			instance_ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
			instance_ci.enabledExtensionCount = extensions_count;
			instance_ci.ppEnabledExtensionNames = extensions;
			check(vkCreateInstance(&instance_ci, mp_allocator, &m_instance));
		}
	}
	void Vulkan::selectPhysicalDevice()
	{
		RZAssertDebug(m_physical_device == VK_NULL_HANDLE, "physical device is already created");

		// select physical device
		uint32_t device_count = 0;
		check(vkEnumeratePhysicalDevices(m_instance, &device_count, nullptr));
		RZAssert(device_count != 0, "failed to find at least one physical device");

		std::unique_ptr<VkPhysicalDevice[], decltype(std::free)*> physical_devices(
			(VkPhysicalDevice*)std::malloc(sizeof(VkPhysicalDevice) * device_count),
			std::free);
		check(vkEnumeratePhysicalDevices(m_instance, &device_count, physical_devices.get()));

		const std::unordered_map<VkPhysicalDeviceType, int32_t> device_type_rank = {
			{VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 1},
			{VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, 2},
			{VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_CPU, 3}
		};
		VkPhysicalDeviceType best_device_type = VK_PHYSICAL_DEVICE_TYPE_CPU;
		uint32_t best_dev_idx = 0;
		for (uint32_t dev_idx = 0; dev_idx < device_count; dev_idx++)
		{
			VkPhysicalDeviceProperties device_properties;
			vkGetPhysicalDeviceProperties(physical_devices[dev_idx], &device_properties);
			if (device_type_rank.contains(device_properties.deviceType) &&
				device_type_rank.at(device_properties.deviceType) < best_device_type)
			{
				best_device_type = device_properties.deviceType;
				best_dev_idx = dev_idx;
			}
		}
		m_physical_device = physical_devices[best_dev_idx];

		// select graphics queue family
		uint32_t queue_count{};
		vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &queue_count, nullptr);
		std::unique_ptr<VkQueueFamilyProperties[], decltype(std::free)*> queues(
			(VkQueueFamilyProperties*)std::malloc(sizeof(VkQueueFamilyProperties) * queue_count),
			std::free);
		vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &queue_count, queues.get());
		for (uint32_t queue_idx = 0; queue_idx < queue_count; queue_idx++)
		{
			if (queues[queue_idx].queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				m_queue_family_idx = queue_idx;
				break;
			}
		}
		RZAssert(m_queue_family_idx != std::numeric_limits<uint32_t>::max(), "Failed to find graphics queue");
	}
	void Vulkan::createLogicalDevice()
	{
		// create queue
		int device_extensions_count = 1;
		const char* device_extensions[] = { "VK_KHR_swapchain" };
		const float queue_priority = 1.0f;
		VkDeviceQueueCreateInfo queue_ci[1] = {};
		queue_ci[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queue_ci[0].queueCount = 1;
		queue_ci[0].queueFamilyIndex = m_queue_family_idx;
		queue_ci[0].pQueuePriorities = &queue_priority;

		// create logical device
		VkDeviceCreateInfo device_ci{};
		device_ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		device_ci.queueCreateInfoCount = sizeof(queue_ci) / sizeof(queue_ci[0]);
		device_ci.pQueueCreateInfos = queue_ci;
		device_ci.enabledExtensionCount = sizeof(device_extensions) / sizeof(device_extensions[0]);
		device_ci.ppEnabledExtensionNames = device_extensions;
		check(vkCreateDevice(m_physical_device, &device_ci, mp_allocator, &m_logical_device));

		// get device queue
		vkGetDeviceQueue(m_logical_device, m_queue_family_idx, 0, &m_queue);
	}
	void Vulkan::createDescriptorPool()
	{
		const uint32_t pool_size = 1024;
		VkDescriptorPoolSize pool_sizes[] =
		{
			{ VK_DESCRIPTOR_TYPE_SAMPLER, pool_size },
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, pool_size },
			{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, pool_size },
			{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, pool_size },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, pool_size },
			{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, pool_size },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, pool_size },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pool_size },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, pool_size },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, pool_size },
			{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, pool_size }
		};
		VkDescriptorPoolCreateInfo pool_ci{};
		pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		pool_ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		pool_ci.maxSets = pool_size * (sizeof(pool_sizes) / sizeof(pool_sizes[0]));
		pool_ci.poolSizeCount = (sizeof(pool_sizes) / sizeof(pool_sizes[0]));
		pool_ci.pPoolSizes = pool_sizes;
		check(vkCreateDescriptorPool(m_logical_device, &pool_ci, mp_allocator, &m_descriptor_pool));
	}
	void Vulkan::createWindowSurface()
	{
		RZAssertDebug(m_window_surface == VK_NULL_HANDLE, "window surface already created");
		m_window_surface = m_glfw.createWindowSurface();
	}

	void Vulkan::createWindow(const int width, const int height)
	{
		m_imgui_main_window.Surface = m_window_surface;

		// Check for WSI support
		VkBool32 res;
		check(vkGetPhysicalDeviceSurfaceSupportKHR(
			m_physical_device,
			m_queue_family_idx,
			m_imgui_main_window.Surface,
			&res));
		RZAssert(res == VK_TRUE, "no WSI support on physical device 0");

		// Select Surface Format		
		m_imgui_main_window.SurfaceFormat = selectSurfaceFormat();

		// Select Present Mode
		#ifdef IMGUI_UNLIMITED_FRAME_RATE
		VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_KHR };
		#else
		VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_FIFO_KHR };
		#endif

		m_imgui_main_window.PresentMode = ImGui_ImplVulkanH_SelectPresentMode(
			m_physical_device,
			m_imgui_main_window.Surface,
			&present_modes[0],
			IM_ARRAYSIZE(present_modes));

		// Create SwapChain, RenderPass, Framebuffer, etc.
		IM_ASSERT(m_min_image_count >= 2);
		ImGui_ImplVulkanH_CreateOrResizeWindow(
			m_instance,
			m_physical_device,
			m_logical_device,
			&m_imgui_main_window,
			m_queue_family_idx,
			mp_allocator,
			width, height,
			m_min_image_count);
	}

	void Vulkan::destroyDescriptorPool()
	{
		if (m_descriptor_pool != VK_NULL_HANDLE)
		{
			vkDestroyDescriptorPool(m_logical_device, m_descriptor_pool, mp_allocator);
			m_descriptor_pool = VK_NULL_HANDLE;
		}
	}
	void Vulkan::destroyLogicalDevice()
	{
		if (m_logical_device != VK_NULL_HANDLE)
		{
			vkDestroyDevice(m_logical_device, mp_allocator);
			m_logical_device = VK_NULL_HANDLE;
		}
	}
	void Vulkan::destroyInstance()
	{
		if (m_debug_report_callback != VK_NULL_HANDLE)
		{
			// destroy debug report callback
			auto vkDestroyDebugReportCallbackEXT =
				(PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(m_instance, "vkDestroyDebugReportCallbackEXT");
			vkDestroyDebugReportCallbackEXT(m_instance, m_debug_report_callback, mp_allocator);
			m_debug_report_callback = VK_NULL_HANDLE;
		}
		if (m_instance != VK_NULL_HANDLE)
		{
			// destroy vulkan instance
			vkDestroyInstance(m_instance, mp_allocator);
			m_instance = VK_NULL_HANDLE;
			mp_allocator = VK_NULL_HANDLE;
		}
	}

	VKAPI_ATTR VkBool32 VKAPI_CALL Vulkan::debugReport(
		[[maybe_unused]] VkDebugReportFlagsEXT flags,
		VkDebugReportObjectTypeEXT objectType,
		[[maybe_unused]] uint64_t object,
		[[maybe_unused]] size_t location,
		[[maybe_unused]] int32_t messageCode,
		[[maybe_unused]] const char* pLayerPrefix,
		const char* pMessage,
		[[maybe_unused]] void* pUserData)
	{
		if constexpr (VULKAN_DEBUG_REPORT)
		{
			std::cout << std::format(
				"[vk debug report] ObjectType: {}\nMessage: {}\n\n",
				int(objectType), pMessage);
		}
		return VK_FALSE;
	}

	VkSurfaceFormatKHR Vulkan::selectSurfaceFormat()
	{
		uint32_t available_format_count = 0;
		check(vkGetPhysicalDeviceSurfaceFormatsKHR(
			m_physical_device, m_window_surface,
			&available_format_count, NULL));
		std::vector<VkSurfaceFormatKHR> available_formats(available_format_count);
		vkGetPhysicalDeviceSurfaceFormatsKHR(
			m_physical_device, m_window_surface,
			&available_format_count, available_formats.data());

		const VkFormat requested_image_formats[] = {
			   VK_FORMAT_B8G8R8A8_UNORM,
			   VK_FORMAT_R8G8B8A8_UNORM,
			   VK_FORMAT_B8G8R8_UNORM,
			   VK_FORMAT_R8G8B8_UNORM };
		const VkColorSpaceKHR requested_color_space = VK_COLORSPACE_SRGB_NONLINEAR_KHR; // required by ImGui

		if (available_formats.size() == 1 && available_formats[0].format == VK_FORMAT_UNDEFINED)
		{
			return VkSurfaceFormatKHR{
				.format = available_formats[0].format,
				.colorSpace = requested_color_space
			};
		}

		for (const auto& requested_format : requested_image_formats)
			for (const auto& available_format : available_formats)
				if (available_format.format == requested_format &&
					available_format.colorSpace == requested_color_space)
					return available_format;

		return available_formats.front();
	}
}
