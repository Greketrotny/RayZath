module;

#include "vulkan/vulkan.h"
#include "vec2.h"
#include "rzexception.h"

#include <vector>
#include <array>

module rz.ui.rendering.vulkan.window;

import rz.ui.rendering.glfw;

namespace RayZath::UI::Rendering::Vulkan
{
	Frame::Frame(
		Instance& instance,
		VkImage back_buffer,
		const VkSurfaceFormatKHR& surface_format,
		VkRenderPass render_pass,
		Math::vec2ui32 resolution)
		: mr_instance(instance)
		, m_back_buffer(back_buffer)
	{
		createBackBufferView(surface_format);
		createFrameBuffer(render_pass, resolution);

		createCommandPool();
		createCommandBuffer();
		createFences();
		createSemaphores();
	}
	Frame::~Frame()
	{
		destroySemaphores();
		destroyFences();
		destroyCommandBuffer();
		destroyCommandPool();
		destroyFrameBuffer();
		destroyBackBufferView();
	}

	void Frame::resetCommandPool()
	{
		check(vkResetCommandPool(mr_instance.logicalDevice(), m_command_pool, 0));
		VkCommandBufferBeginInfo info{};
		info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		check(vkBeginCommandBuffer(m_command_buffer, &info));
	}

	void Frame::createBackBufferView(const VkSurfaceFormatKHR& surface_format)
	{
		VkImageSubresourceRange image_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

		VkImageViewCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		info.viewType = VK_IMAGE_VIEW_TYPE_2D;
		info.format = surface_format.format;
		info.components.r = VK_COMPONENT_SWIZZLE_R;
		info.components.g = VK_COMPONENT_SWIZZLE_G;
		info.components.b = VK_COMPONENT_SWIZZLE_B;
		info.components.a = VK_COMPONENT_SWIZZLE_A;
		info.subresourceRange = image_range;
		info.image = m_back_buffer;

		check(vkCreateImageView(
			mr_instance.logicalDevice(),
			&info,
			mr_instance.allocator(),
			&m_back_buffer_view));
	}
	void Frame::createFrameBuffer(
		VkRenderPass render_pass,
		Math::vec2ui32 resolution)
	{
		VkImageView attachment[1]{};
		attachment[0] = m_back_buffer_view;

		VkFramebufferCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		info.renderPass = render_pass;
		info.attachmentCount = 1;
		info.pAttachments = attachment;
		info.width = resolution.x;
		info.height = resolution.y;
		info.layers = 1;

		check(vkCreateFramebuffer(
			mr_instance.logicalDevice(),
			&info,
			mr_instance.allocator(),
			&m_frame_buffer));
	}
	void Frame::createCommandPool()
	{
		VkCommandPoolCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		info.queueFamilyIndex = mr_instance.queueFamilyIdx();
		check(vkCreateCommandPool(
			mr_instance.logicalDevice(),
			&info,
			mr_instance.allocator(),
			&m_command_pool));
	}
	void Frame::createCommandBuffer()
	{
		VkCommandBufferAllocateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		info.commandPool = m_command_pool;
		info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		info.commandBufferCount = 1;

		check(vkAllocateCommandBuffers(mr_instance.logicalDevice(), &info, &m_command_buffer));
	}
	void Frame::createFences()
	{
		VkFenceCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
		check(vkCreateFence(mr_instance.logicalDevice(), &info, mr_instance.allocator(), &m_fence));
	}
	void Frame::createSemaphores()
	{
		VkSemaphoreCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		check(vkCreateSemaphore(
			mr_instance.logicalDevice(),
			&info,
			mr_instance.allocator(),
			&m_semaphore.image_acquired));
		check(vkCreateSemaphore(
			mr_instance.logicalDevice(),
			&info,
			mr_instance.allocator(),
			&m_semaphore.render_complete));
	}

	void Frame::destroySemaphores()
	{
		if (m_semaphore.image_acquired)
		{
			vkDestroySemaphore(mr_instance.logicalDevice(), m_semaphore.image_acquired, mr_instance.allocator());
			m_semaphore.image_acquired = VK_NULL_HANDLE;
		}
		if (m_semaphore.render_complete)
		{
			vkDestroySemaphore(mr_instance.logicalDevice(), m_semaphore.render_complete, mr_instance.allocator());
			m_semaphore.render_complete = VK_NULL_HANDLE;
		}
	}
	void Frame::destroyFences()
	{
		vkDestroyFence(mr_instance.logicalDevice(), m_fence, mr_instance.allocator());
	}
	void Frame::destroyCommandBuffer()
	{
		if (m_command_buffer)
		{
			vkFreeCommandBuffers(mr_instance.logicalDevice(), m_command_pool, 1, &m_command_buffer);
			m_command_buffer = VK_NULL_HANDLE;
		}

	}
	void Frame::destroyCommandPool()
	{
		if (m_command_pool)
		{
			vkDestroyCommandPool(
				mr_instance.logicalDevice(),
				m_command_pool,
				mr_instance.allocator());
			m_command_pool = VK_NULL_HANDLE;
		}
	}
	void Frame::destroyFrameBuffer()
	{
		vkDestroyFramebuffer(
			mr_instance.logicalDevice(),
			m_frame_buffer,
			mr_instance.allocator());
	}
	void Frame::destroyBackBufferView()
	{
		vkDestroyImageView(
			mr_instance.logicalDevice(),
			m_back_buffer_view,
			mr_instance.allocator());
	}


	// -------- Window --------
	Window::Window(Instance& instance)
		: mr_instance(instance)
	{}
	Window::~Window()
	{
		vkQueueWaitIdle(mr_instance.queue());

		destroyRenderPass();
		m_frame.clear();
		vkDestroySwapchainKHR(mr_instance.logicalDevice(), m_swapchain, mr_instance.allocator());
		vkDestroySurfaceKHR(mr_instance.vulkanInstance(), m_surface, mr_instance.allocator());
	}

	void Window::init(RayZath::UI::Rendering::GLFW::GLFWWrapper& glfw, const Math::vec2ui32 resolution)
	{
		m_resolution = resolution;

		// create window surface
		m_surface = glfw.createWindowSurface(mr_instance.vulkanInstance(), mr_instance.allocator());

		// check for WSI support
		VkBool32 supported;
		check(vkGetPhysicalDeviceSurfaceSupportKHR(
			mr_instance.physicalDevice(),
			mr_instance.queueFamilyIdx(),
			m_surface,
			&supported));
		RZAssert(supported == VK_TRUE, "no WSI support on physical device 0");

		m_surface_format = selectSurfaceFormat();
		m_present_mode = selectPresentMode();

		check(vkDeviceWaitIdle(mr_instance.logicalDevice()));

		createSwapChain(resolution);
		createRenderPass();
		createFrames();
	}

	void Window::reset(const Math::vec2ui32 resolution)
	{
		vkDeviceWaitIdle(mr_instance.logicalDevice());

		m_resolution = resolution;
		m_rebuild = false;

		m_frame.clear();
		createSwapChain(resolution);
		createRenderPass();
		createFrames();

		m_frame_index = 0;
	}

	void Window::acquireNextImage()
	{
		auto err = vkAcquireNextImageKHR(
			mr_instance.logicalDevice(),
			m_swapchain,
			UINT64_MAX,
			m_frame[m_semaphore_index].m_semaphore.image_acquired,
			VK_NULL_HANDLE,
			&m_frame_index);
		if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
		{
			m_rebuild = true;
			return;
		}
		check(err);
	}
	void Window::waitForFence()
	{
		VkFence fence[1]{ currentFrame().m_fence  };
		check(vkWaitForFences(mr_instance.logicalDevice(), 1, fence, VK_TRUE, UINT64_MAX));
		check(vkResetFences(mr_instance.logicalDevice(), 1, fence));
	}
	void Window::incrementSemaphoreIndex()
	{
		m_semaphore_index = (size_t(m_semaphore_index) + 1) % m_frame.size();
	}

	void Window::resetCommandPool()
	{
		acquireNextImage();
		waitForFence();

		currentFrame().resetCommandPool();
	}
	void Window::beginRenderPass()
	{
		{
			VkClearColorValue clearColorValue;
			clearColorValue.float32[0] = 0.2f;
			clearColorValue.float32[1] = 1.0f;
			clearColorValue.float32[2] = 0.2f;
			clearColorValue.float32[3] = 0.5f;
			VkClearValue clearValue{};
			clearValue.color = clearColorValue;
			VkRenderPassBeginInfo info{};
			info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			info.renderPass = renderPass();
			info.framebuffer = currentFrame().m_frame_buffer;
			info.renderArea.extent.width = resolution().x;
			info.renderArea.extent.height = resolution().y;
			info.clearValueCount = 1;
			info.pClearValues = &clearValue;
			vkCmdBeginRenderPass(
				currentFrame().commandBuffer(),
				&info,
				VK_SUBPASS_CONTENTS_INLINE);
		}		
	}
	void Window::endRenderPass()
	{
		VkSemaphore image_acquired_semaphore = frame(semaphoreIndex()).m_semaphore.image_acquired;
		VkSemaphore render_complete_semaphore = frame(semaphoreIndex()).m_semaphore.render_complete;

		vkCmdEndRenderPass(currentFrame().commandBuffer());
		{
			VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			VkSubmitInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			info.waitSemaphoreCount = 1;
			info.pWaitSemaphores = &image_acquired_semaphore;
			info.pWaitDstStageMask = &wait_stage;
			info.commandBufferCount = 1;
			info.pCommandBuffers = &currentFrame().commandBuffer();
			info.signalSemaphoreCount = 1;
			info.pSignalSemaphores = &render_complete_semaphore;

			check(vkEndCommandBuffer(currentFrame().commandBuffer()));
			check(vkQueueSubmit(mr_instance.queue(), 1, &info, currentFrame().m_fence));
		}
	}

	void Window::framePresent()
	{
		if (m_rebuild) return;

		VkSemaphore render_complete_semaphore = frame(semaphoreIndex()).m_semaphore.render_complete;

		auto image_index = frameIndex();
		auto swapchain_object = swapchain();
		VkPresentInfoKHR info{};
		info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		info.waitSemaphoreCount = 1;
		info.pWaitSemaphores = &render_complete_semaphore;
		info.swapchainCount = 1;
		info.pSwapchains = &swapchain_object;
		info.pImageIndices = &image_index;
		const VkResult err = vkQueuePresentKHR(mr_instance.queue(), &info);
		if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
		{
			m_rebuild = true;
			return;
		}
		check(err);

		incrementSemaphoreIndex();
	}

	void Window::createSwapChain(const Math::vec2ui32 resolution)
	{
		auto old_swapchain = m_swapchain;

		VkSwapchainCreateInfoKHR info{};
		info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		info.surface = m_surface;
		info.minImageCount = m_min_image_count;
		info.imageFormat = m_surface_format.format;
		info.imageColorSpace = m_surface_format.colorSpace;
		info.imageArrayLayers = 1;
		info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		info.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
		info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		info.presentMode = m_present_mode;
		info.clipped = VK_TRUE;
		info.oldSwapchain = old_swapchain;

		VkSurfaceCapabilitiesKHR surface_capabs;
		check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
			mr_instance.physicalDevice(), m_surface, &surface_capabs));
		if (info.minImageCount < surface_capabs.minImageCount)
			info.minImageCount = surface_capabs.minImageCount;
		else if (
			surface_capabs.maxImageCount != 0 &&
			info.minImageCount > surface_capabs.maxImageCount)
			info.minImageCount = surface_capabs.maxImageCount;

		if (surface_capabs.currentExtent.width == 0xffffffff)
		{
			info.imageExtent.width = m_resolution.x = resolution.x;
			info.imageExtent.height = m_resolution.y = resolution.y;
		}
		else
		{
			info.imageExtent.width = m_resolution.x = surface_capabs.currentExtent.width;
			info.imageExtent.height = m_resolution.y = surface_capabs.currentExtent.height;
		}
		check(vkCreateSwapchainKHR(
			mr_instance.logicalDevice(),
			&info,
			mr_instance.allocator(),
			&m_swapchain));

		if (old_swapchain)
			vkDestroySwapchainKHR(
				mr_instance.logicalDevice(),
				old_swapchain,
				mr_instance.allocator());
	}
	void Window::createFrames()
	{
		// obtain swap chain back buffers
		uint32_t swapchain_image_count{};
		check(vkGetSwapchainImagesKHR(
			mr_instance.logicalDevice(),
			m_swapchain,
			&swapchain_image_count,
			NULL));
		std::vector<VkImage> back_buffers(swapchain_image_count);
		RZAssert(swapchain_image_count >= m_min_image_count, "swapchain image count too small");
		check(vkGetSwapchainImagesKHR(
			mr_instance.logicalDevice(),
			m_swapchain,
			&swapchain_image_count,
			back_buffers.data()));

		m_frame.clear();
		for (auto back_buffer : back_buffers)
		{
			m_frame.emplace_back(
				mr_instance,
				back_buffer,
				m_surface_format,
				m_render_pass,
				m_resolution);
		}
	}
	void Window::createRenderPass()
	{
		destroyRenderPass();

		VkAttachmentDescription attachment{};
		attachment.format = m_surface_format.format;
		attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference color_attachment{};
		color_attachment.attachment = 0;
		color_attachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &color_attachment;

		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		VkRenderPassCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		info.attachmentCount = 1;
		info.pAttachments = &attachment;
		info.subpassCount = 1;
		info.pSubpasses = &subpass;
		info.dependencyCount = 1;
		info.pDependencies = &dependency;
		check(vkCreateRenderPass(
			mr_instance.logicalDevice(),
			&info,
			mr_instance.allocator(),
			&m_render_pass));
	}

	void Window::destroyRenderPass()
	{
		if (m_render_pass)
		{
			vkDestroyRenderPass(mr_instance.logicalDevice(), m_render_pass, mr_instance.allocator());
			m_render_pass = VK_NULL_HANDLE;
		}
	}

	VkSurfaceFormatKHR Window::selectSurfaceFormat()
	{
		uint32_t available_count = 0;
		Vulkan::check(vkGetPhysicalDeviceSurfaceFormatsKHR(
			mr_instance.physicalDevice(), m_surface,
			&available_count, NULL));
		std::vector<VkSurfaceFormatKHR> available_formats(available_count);
		vkGetPhysicalDeviceSurfaceFormatsKHR(
			mr_instance.physicalDevice(), m_surface,
			&available_count, available_formats.data());

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
	VkPresentModeKHR Window::selectPresentMode()
	{
		#ifdef IMGUI_UNLIMITED_FRAME_RATE
		std::array requested_modes = {
			VK_PRESENT_MODE_MAILBOX_KHR,
			VK_PRESENT_MODE_IMMEDIATE_KHR,
			VK_PRESENT_MODE_FIFO_KHR };
		#else
		std::array requested_modes = { VK_PRESENT_MODE_FIFO_KHR };
		#endif

		uint32_t available_count = 0;
		Vulkan::check(vkGetPhysicalDeviceSurfacePresentModesKHR(
			mr_instance.physicalDevice(), m_surface,
			&available_count, NULL));
		std::vector<VkPresentModeKHR> available_modes(available_count);
		vkGetPhysicalDeviceSurfacePresentModesKHR(
			mr_instance.physicalDevice(), m_surface,
			&available_count, available_modes.data());

		for (const auto& requested_mode : requested_modes)
			for (const auto& available_mode : available_modes)
				if (requested_mode == available_mode)
					return requested_mode;

		return VK_PRESENT_MODE_FIFO_KHR; // mandatory
}
}
