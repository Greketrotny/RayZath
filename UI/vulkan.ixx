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
	class Vulkan
	{
	public:
		VkInstance m_vk_instance = VK_NULL_HANDLE;
		VkAllocationCallbacks* mp_vk_allocator = VK_NULL_HANDLE;
		VkDebugReportCallbackEXT m_vk_debug_report_callback = VK_NULL_HANDLE;

		VkPhysicalDevice m_physical_device = VK_NULL_HANDLE;
		uint32_t m_queue_family_idx = std::numeric_limits<uint32_t>::max();
		VkDevice m_logical_device = VK_NULL_HANDLE;
		VkQueue m_queue = VK_NULL_HANDLE;

	public:
		~Vulkan();

		static VkResult check(VkResult result);

		void createInstance(const char** extensions, const uint32_t extensions_count);
		void selectPhysicalDevice();
		void createLogicalDevice();

		void destroyLogicalDevice();
		void destroyInstance();

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
