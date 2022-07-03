#pragma once

#include "vulkan/vulkan.h"

#include <concepts>
#include <utility>

#include "glfw.hpp"

namespace RayZath::UI::Rendering::Vulkan
{
	inline constexpr bool VULKAN_DEBUG_REPORT = true;

	template <class T>
	concept pointer = std::is_pointer_v<T>;
	template <pointer T>
	class Handle
	{
	private:
		using value_t = std::remove_pointer_t<T>;
		value_t* m_object;

	public:
		Handle()
			: m_object(nullptr)
		{}
		Handle(const Handle& other) = delete;
		Handle(Handle&& other) noexcept
			: m_object(std::exchange(other.m_object, nullptr))
		{}
		Handle(value_t* object)
			: m_object(object)
		{}

		Handle& operator=(const Handle& other) = delete;
		Handle& operator=(Handle&& other) noexcept
		{
			m_object = std::exchange(other.m_object, nullptr);
			return *this;
		}
		
		explicit operator bool()
		{
			return m_object != nullptr;
		}
		value_t& operator->()
		{
			return *m_object;
		}
		value_t** operator&()
		{
			return &m_object;
		}
		operator value_t* () const
		{
			return m_object;
		}
	};

	class Instance
	{
	private:
		static Instance sm_instance;
		VkInstance m_instance = VK_NULL_HANDLE;
		VkAllocationCallbacks* mp_allocator = VK_NULL_HANDLE;
		VkDebugReportCallbackEXT m_debug_report_callback = VK_NULL_HANDLE;

		VkPhysicalDevice m_physical_device = VK_NULL_HANDLE;
		uint32_t m_queue_family_idx = std::numeric_limits<uint32_t>::max();
		VkDevice m_logical_device = VK_NULL_HANDLE;
		VkQueue m_queue = VK_NULL_HANDLE;
		VkDescriptorPool m_descriptor_pool = VK_NULL_HANDLE;

	public:
		Instance(const Instance&) = delete;
		Instance(Instance&&) = delete;
	private:
		Instance() = default;
	public:
		~Instance();

		Instance& operator=(const Instance&) = delete;
		Instance& operator=(Instance&&) = delete;

		static Instance& get() { return sm_instance; };
			
		auto vulkanInstance() { return m_instance; }
		auto* allocator() { return mp_allocator; }
		auto physicalDevice() { return m_physical_device; }
		auto logicalDevice() { return m_logical_device; }
		auto queueFamilyIdx() { return m_queue_family_idx; }
		auto queue() { return m_queue; }
		auto descriptorPool() { return m_descriptor_pool; }

		void init(GLFW::Module& glfw);
		void destroy();
	private:
		void createVulkanInstance(GLFW::Module& glfw);
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

	VkResult check(VkResult result);
}
