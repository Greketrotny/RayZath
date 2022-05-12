module;

#include "vulkan/vulkan.h"
#include <iostream>
#include <type_traits>
#include <format>
#include <unordered_map>

export module rz.ui.rendering.vulkan;

export inline constexpr bool VULKAN_DEBUG_REPORT = true;

export namespace RayZath::UI::Render
{
	class GLFW; 

	class Vulkan
	{
	public:
		GLFW& m_glfw;
		VkInstance m_instance = VK_NULL_HANDLE;
		VkAllocationCallbacks* mp_allocator = VK_NULL_HANDLE;
		VkDebugReportCallbackEXT m_debug_report_callback = VK_NULL_HANDLE;

		VkPhysicalDevice m_physical_device = VK_NULL_HANDLE;
		uint32_t m_queue_family_idx = std::numeric_limits<uint32_t>::max();
		VkDevice m_logical_device = VK_NULL_HANDLE;
		VkQueue m_queue = VK_NULL_HANDLE;

		VkDescriptorPool m_descriptor_pool = VK_NULL_HANDLE;

		VkSurfaceKHR m_window_surface = VK_NULL_HANDLE;

	public:
		Vulkan(GLFW&);
		~Vulkan();

		static VkResult check(VkResult result);

		void init();
	private:
		void createInstance();
		void selectPhysicalDevice();
		void createLogicalDevice();
		void createDescriptorPool();
		void createWindowSurface();

	public:
		void destroyDescriptorPool();
		void destroyLogicalDevice();
		void destroyInstance();

	private:
		static VKAPI_ATTR VkBool32 VKAPI_CALL debugReport(
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
	};
}
