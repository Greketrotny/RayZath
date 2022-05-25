module;

#include "vulkan/vulkan.h"
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include "glfw3.h"

#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "vec2.h"
#include "bitmap.h"

#include <iostream>
#include <string>
#include <limits>
#include <format>
#include <unordered_map>
#include <memory>

export module rz.ui.rendering.backend;

export inline constexpr bool VULKAN_DEBUG_REPORT = true;

export namespace RayZath::UI::Render
{
	class GLFW;
	class Vulkan;

	class GLFW
	{
	private:
		Vulkan& m_vulkan;

		GLFWwindow* mp_window = nullptr;
		uint32_t m_extensions_count = 0;
		const char** m_extensions = nullptr;

	public:
		GLFW(Vulkan& vulkan);
		~GLFW();

	public:
		GLFWwindow* window() { return mp_window; }

		const char** extensions() { return m_extensions; }
		auto extensionsCount() { return m_extensions_count; }
		Math::vec2ui32 frameBufferSize();
		Math::vec2ui32 windowSize();

		void init();

		VkSurfaceKHR createWindowSurface();
	};

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

	public:
		static constexpr int m_min_image_count = 2;
		bool m_swapchain_rebuild = false;

		ImGui_ImplVulkanH_Window m_imgui_main_window;

		// --------------------
		uint32_t image_width, image_height;
		VkImage m_image = VK_NULL_HANDLE;
		VkImageView m_image_view = VK_NULL_HANDLE;
		VkSampler m_sampler = VK_NULL_HANDLE;
		VkDescriptorSetLayout m_descriptor_set_layout = VK_NULL_HANDLE;
		VkDeviceMemory m_device_memory = VK_NULL_HANDLE;
		ImTextureID m_render_texture = nullptr;

		VkBuffer m_buffer = VK_NULL_HANDLE;
		VkDeviceMemory m_staging_memory = VK_NULL_HANDLE;

	public:
		Vulkan(GLFW& glfw);
		~Vulkan();

		static VkResult check(VkResult result);

		void init();
		void destroy();
	private:
		void createInstance();
		void selectPhysicalDevice();
		void createLogicalDevice();
		void createDescriptorPool();
		void createWindowSurface();
	public:
		void createWindow(const Math::vec2ui32& frame_buffer_size);

		void createSwapChain(const Math::vec2ui32& frame_buffer_size);
		void createCommandBuffers();


	private:
		void destroyWindow();
		void destroyDescriptorPool();
		void destroyLogicalDevice();
		void destroyInstance();

	public:
		void frameRender();
		void framePresent();

		void createImage(const Graphics::Bitmap& bitmap, VkCommandBuffer command_buffer);
		void updateImage(const Graphics::Bitmap& bitmap, VkCommandBuffer command_buffer);
		void destroyImage();

	private:
		static VKAPI_ATTR VkBool32 VKAPI_CALL debugReport(
			[[maybe_unused]] VkDebugReportFlagsEXT flags,
			VkDebugReportObjectTypeEXT objectType,
			[[maybe_unused]] uint64_t object,
			[[maybe_unused]] size_t location,
			[[maybe_unused]] int32_t messageCode,
			[[maybe_unused]] const char* pLayerPrefix,
			const char* pMessage,
			[[maybe_unused]] void* pUserData);

		VkSurfaceFormatKHR selectSurfaceFormat();
		VkPresentModeKHR selectPresentMode();

		void createBuffer(
			VkDeviceSize size,
			VkBufferUsageFlags usage_flags,
			VkMemoryPropertyFlags properties,
			VkBuffer& buffer, VkDeviceMemory& buffer_memory);
		uint32_t findMemoryType(const uint32_t type_filter, const VkMemoryPropertyFlags properties);
	};
}
