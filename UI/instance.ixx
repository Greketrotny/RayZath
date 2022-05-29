export module rz.ui.rendering.vulkan.instance;

import rz.ui.rendering.glfw;

import "vulkan/vulkan.h";

import <limits>;

export inline constexpr bool VULKAN_DEBUG_REPORT = true;

namespace RayZath::UI::Rendering::Vulkan
{
	export class Instance
	{
	private:
	public:
		VkInstance m_instance = VK_NULL_HANDLE;
		VkAllocationCallbacks* mp_allocator = VK_NULL_HANDLE;
		VkDebugReportCallbackEXT m_debug_report_callback = VK_NULL_HANDLE;

		VkPhysicalDevice m_physical_device = VK_NULL_HANDLE;
		uint32_t m_queue_family_idx = std::numeric_limits<uint32_t>::max();
		VkDevice m_logical_device = VK_NULL_HANDLE;
		VkQueue m_queue = VK_NULL_HANDLE;
		VkDescriptorPool m_descriptor_pool = VK_NULL_HANDLE;

	public:
		~Instance();

		void init(GLFW::GLFWWrapper& glfw);
		void destroy();

	private:
		void createVulkanInstance(GLFW::GLFWWrapper& glfw);
		void selectPhysicalDevice();
		void createLogicalDevice();
		void createDescriptorPool();

		void destroyDescriptorPool();
		void destroyLogicalDevice();
		void destroyVulkanInstance();

		static VKAPI_ATTR VkBool32 VKAPI_CALL debugReport(
			[[maybe_unused]] VkDebugReportFlagsEXT flags,
			VkDebugReportObjectTypeEXT objectType,
			[[maybe_unused]] uint64_t object,
			[[maybe_unused]] size_t location,
			[[maybe_unused]] int32_t messageCode,
			[[maybe_unused]] const char* pLayerPrefix,
			const char* pMessage,
			[[maybe_unused]] void* pUserData);
	};

	export VkResult check(VkResult result);
}
