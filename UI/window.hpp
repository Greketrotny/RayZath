#pragma once

#include "instance.hpp"

#include "vec2.h"
#include "rzexception.h"

#include <vector>

namespace RayZath::UI::Rendering::Vulkan
{
	struct Frame
	{
		struct Semaphores
		{
			Handle<VkSemaphore> image_acquired = VK_NULL_HANDLE;
			Handle<VkSemaphore> render_complete = VK_NULL_HANDLE;
		};

	private:
	public:
		Handle<VkImage> m_back_buffer = VK_NULL_HANDLE;
		Handle<VkImageView> m_back_buffer_view = VK_NULL_HANDLE;
		Handle<VkFramebuffer> m_frame_buffer = VK_NULL_HANDLE;
		Handle<VkCommandPool> m_command_pool = VK_NULL_HANDLE;
		Handle<VkCommandBuffer> m_command_buffer = VK_NULL_HANDLE;
		Handle<VkFence> m_render_complete = VK_NULL_HANDLE;
		Semaphores m_semaphore;

	public:
		Frame(Frame&&) = default;
		Frame(const Frame&) = delete;
		Frame(
			VkImage back_buffer,
			const VkSurfaceFormatKHR& surface_format,
			VkRenderPass render_pass,
			Math::vec2ui32 resolution);
		~Frame();

		Frame& operator=(const Frame&) = delete;
		Frame& operator=(Frame&&) = default;

		auto& commandBuffer() { return m_command_buffer; }

		void resetCommandPool();

	private:
		void createBackBufferView(const VkSurfaceFormatKHR& surface_format);
		void createFrameBuffer(VkRenderPass render_pass, Math::vec2ui32 resolution);
		void createCommandPool();
		void createCommandBuffer();
		void createFences();
		void createSemaphores();

		void destroySemaphores();
		void destroyFences();
		void destroyCommandBuffer();
		void destroyCommandPool();
		void destroyFrameBuffer();
		void destroyBackBufferView();
	};

	class Window
	{
	private:
		Math::vec2ui32 m_resolution{};

		VkSurfaceKHR m_surface = VK_NULL_HANDLE;
		VkSurfaceFormatKHR m_surface_format{};
		VkPresentModeKHR m_present_mode{};
		VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
		VkRenderPass m_render_pass = VK_NULL_HANDLE;

	public:
		static constexpr int m_min_image_count = 2;
	private:
		bool m_rebuild = false;
		uint32_t m_frame_index = 0;
		uint32_t m_semaphore_index = 0;
		std::vector<Frame> m_frame;

	public:
		Window(const Window& other) = delete;
		Window(Window&& other) = delete;
		Window() = default;
		~Window();

		Window& operator=(const Window& other) = delete;
		Window& operator=(Window&& other) = delete;

		auto resolution() { return m_resolution; }
		auto swapchain() { return m_swapchain; }
		auto renderPass() { return m_render_pass; }
		auto rebuild() { return m_rebuild; }
		auto& frame(uint32_t index) { return m_frame[index]; }
		auto& currentFrame() { return m_frame[m_frame_index]; }
		auto frameIndex() { return m_frame_index; }
		auto semaphoreIndex() { return m_semaphore_index; }


		void init(RayZath::UI::Rendering::GLFW::Module& glfw, const Math::vec2ui32 resolution);
		void reset(const Math::vec2ui32 resolution);

		void acquireNextImage();
		void waitForFence();
		void incrementSemaphoreIndex();

		void resetCommandPool();
		void beginRenderPass();
		void endRenderPass();

		void framePresent();

	private:
		void createSwapChain(const Math::vec2ui32 resolution);
		void createFrames();
		void createRenderPass();

		void destroyRenderPass();

		VkSurfaceFormatKHR selectSurfaceFormat();
		VkPresentModeKHR selectPresentMode();
	};
}
