module;

#include "vulkan/vulkan.h"
#include "imgui_impl_vulkan.h"

#include "vec2.h"
#include "rzexception.h"
#include "bitmap.h"

#include <iostream>
#include <string>
#include <limits>
#include <format>
#include <unordered_map>
#include <memory>
#include <array>

module rz.ui.rendering.vulkan;

namespace RayZath::UI::Rendering::Vulkan
{
	VkResult VulkanWrapper::check(VkResult result)
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

	VulkanWrapper::VulkanWrapper(GLFW::GLFWWrapper& glfw)
		: m_glfw(glfw)
	{}
	VulkanWrapper::~VulkanWrapper()
	{
		destroy();
	}
	
	void VulkanWrapper::init()
	{
		m_instance.init(m_glfw);

		createWindowSurface();
	}
	void VulkanWrapper::destroy()
	{
		destroyImage();
		// ------------------
		destroyWindow();

		m_instance.destroy();
	}
	
	
	void VulkanWrapper::createWindowSurface()
	{
		RZAssertDebug(m_imgui_main_window.Surface == VK_NULL_HANDLE, "window surface already created");
		m_imgui_main_window.Surface = m_glfw.createWindowSurface(
		m_instance.m_instance, m_instance.mp_allocator);
	}

	void VulkanWrapper::createWindow(const Math::vec2ui32& frame_buffer_size)
	{
		// Check for WSI support
		VkBool32 res;
		check(vkGetPhysicalDeviceSurfaceSupportKHR(
			m_instance.m_physical_device,
			m_instance.m_queue_family_idx,
			m_imgui_main_window.Surface,
			&res));
		RZAssert(res == VK_TRUE, "no WSI support on physical device 0");

		// Select Surface Format		
		m_imgui_main_window.SurfaceFormat = selectSurfaceFormat();

		// Select Present Mode
		m_imgui_main_window.PresentMode = selectPresentMode();

		// Create SwapChain, RenderPass, Framebuffer, etc.
		createSwapChain(frame_buffer_size);
		createCommandBuffers();
	}
	void VulkanWrapper::destroyWindow()
	{
		ImGui_ImplVulkanH_DestroyWindow(
			m_instance.m_instance,
			m_instance.m_logical_device,
			&m_imgui_main_window,
			m_instance.mp_allocator);
	}

	void VulkanWrapper::createSwapChain(const Math::vec2ui32& frame_buffer_size)
	{
		VkSwapchainKHR old_swapchain = m_imgui_main_window.Swapchain;
		m_imgui_main_window.Swapchain = VK_NULL_HANDLE;
		check(vkDeviceWaitIdle(m_instance.m_logical_device));

		// Destroy old Framebuffer
		for (uint32_t i = 0; i < m_imgui_main_window.ImageCount; i++)
		{
			auto& frame = m_imgui_main_window.Frames[i];
			auto& semaphore = m_imgui_main_window.FrameSemaphores[i];

			// destroy frame
			vkDestroyFence(m_instance.m_logical_device, frame.Fence, m_instance.mp_allocator);
			vkFreeCommandBuffers(m_instance.m_logical_device, frame.CommandPool, 1, &frame.CommandBuffer);
			vkDestroyCommandPool(m_instance.m_logical_device, frame.CommandPool, m_instance.mp_allocator);
			frame.Fence = VK_NULL_HANDLE;
			frame.CommandBuffer = VK_NULL_HANDLE;
			frame.CommandPool = VK_NULL_HANDLE;
			vkDestroyImageView(m_instance.m_logical_device, frame.BackbufferView, m_instance.mp_allocator);
			vkDestroyFramebuffer(m_instance.m_logical_device, frame.Framebuffer, m_instance.mp_allocator);

			// destroy semaphore
			vkDestroySemaphore(m_instance.m_logical_device, semaphore.ImageAcquiredSemaphore, m_instance.mp_allocator);
			vkDestroySemaphore(m_instance.m_logical_device, semaphore.RenderCompleteSemaphore, m_instance.mp_allocator);
			semaphore.ImageAcquiredSemaphore = semaphore.RenderCompleteSemaphore = VK_NULL_HANDLE;
		}

		IM_FREE(m_imgui_main_window.Frames);
		IM_FREE(m_imgui_main_window.FrameSemaphores);
		m_imgui_main_window.Frames = NULL;
		m_imgui_main_window.FrameSemaphores = NULL;
		m_imgui_main_window.ImageCount = 0;
		if (m_imgui_main_window.RenderPass)
			vkDestroyRenderPass(m_instance.m_logical_device, m_imgui_main_window.RenderPass, m_instance.mp_allocator);

		// Create Swapchain
		{
			VkSwapchainCreateInfoKHR swapchain_ci{};
			swapchain_ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
			swapchain_ci.surface = m_imgui_main_window.Surface;
			swapchain_ci.minImageCount = m_min_image_count;
			swapchain_ci.imageFormat = m_imgui_main_window.SurfaceFormat.format;
			swapchain_ci.imageColorSpace = m_imgui_main_window.SurfaceFormat.colorSpace;
			swapchain_ci.imageArrayLayers = 1;
			swapchain_ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
			swapchain_ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			swapchain_ci.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
			swapchain_ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
			swapchain_ci.presentMode = m_imgui_main_window.PresentMode;
			swapchain_ci.clipped = VK_TRUE;
			swapchain_ci.oldSwapchain = old_swapchain;

			VkSurfaceCapabilitiesKHR surface_capabs;
			check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
				m_instance.m_physical_device, m_imgui_main_window.Surface, &surface_capabs));
			if (swapchain_ci.minImageCount < surface_capabs.minImageCount)
				swapchain_ci.minImageCount = surface_capabs.minImageCount;
			else if (
				surface_capabs.maxImageCount != 0 &&
				swapchain_ci.minImageCount > surface_capabs.maxImageCount)
				swapchain_ci.minImageCount = surface_capabs.maxImageCount;

			if (surface_capabs.currentExtent.width == 0xffffffff)
			{
				swapchain_ci.imageExtent.width = m_imgui_main_window.Width = frame_buffer_size.x;
				swapchain_ci.imageExtent.height = m_imgui_main_window.Height = frame_buffer_size.y;
			}
			else
			{
				swapchain_ci.imageExtent.width = m_imgui_main_window.Width = surface_capabs.currentExtent.width;
				swapchain_ci.imageExtent.height = m_imgui_main_window.Height = surface_capabs.currentExtent.height;
			}
			check(vkCreateSwapchainKHR(
				m_instance.m_logical_device, &swapchain_ci, m_instance.mp_allocator, &m_imgui_main_window.Swapchain));
			check(vkGetSwapchainImagesKHR(
				m_instance.m_logical_device, m_imgui_main_window.Swapchain, &m_imgui_main_window.ImageCount, NULL));
			VkImage backbuffers[16]{};
			IM_ASSERT(m_imgui_main_window.ImageCount >= m_min_image_count);
			IM_ASSERT(m_imgui_main_window.ImageCount < IM_ARRAYSIZE(backbuffers));
			check(vkGetSwapchainImagesKHR(
				m_instance.m_logical_device, m_imgui_main_window.Swapchain, &m_imgui_main_window.ImageCount, backbuffers));

			IM_ASSERT(m_imgui_main_window.Frames == NULL);
			m_imgui_main_window.Frames =
				(ImGui_ImplVulkanH_Frame*)IM_ALLOC(sizeof(ImGui_ImplVulkanH_Frame) * m_imgui_main_window.ImageCount);
			m_imgui_main_window.FrameSemaphores =
				(ImGui_ImplVulkanH_FrameSemaphores*)IM_ALLOC(sizeof(ImGui_ImplVulkanH_FrameSemaphores) * m_imgui_main_window.ImageCount);
			std::memset(m_imgui_main_window.Frames, 0, sizeof(m_imgui_main_window.Frames[0]) * m_imgui_main_window.ImageCount);
			std::memset(m_imgui_main_window.FrameSemaphores, 0, sizeof(m_imgui_main_window.FrameSemaphores[0]) * m_imgui_main_window.ImageCount);
			for (uint32_t i = 0; i < m_imgui_main_window.ImageCount; i++)
				m_imgui_main_window.Frames[i].Backbuffer = backbuffers[i];
		}
		if (old_swapchain)
			vkDestroySwapchainKHR(m_instance.m_logical_device, old_swapchain, m_instance.mp_allocator);

		// Create the Render Pass
		{
			VkAttachmentDescription attachment{};
			attachment.format = m_imgui_main_window.SurfaceFormat.format;
			attachment.samples = VK_SAMPLE_COUNT_1_BIT;
			attachment.loadOp = m_imgui_main_window.ClearEnable ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
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
			check(vkCreateRenderPass(m_instance.m_logical_device, &info, m_instance.mp_allocator, &m_imgui_main_window.RenderPass));
		}

		// Create The Image Views
		{
			VkImageSubresourceRange image_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

			VkImageViewCreateInfo info{};
			info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			info.viewType = VK_IMAGE_VIEW_TYPE_2D;
			info.format = m_imgui_main_window.SurfaceFormat.format;
			info.components.r = VK_COMPONENT_SWIZZLE_R;
			info.components.g = VK_COMPONENT_SWIZZLE_G;
			info.components.b = VK_COMPONENT_SWIZZLE_B;
			info.components.a = VK_COMPONENT_SWIZZLE_A;
			info.subresourceRange = image_range;

			for (uint32_t i = 0; i < m_imgui_main_window.ImageCount; i++)
			{
				ImGui_ImplVulkanH_Frame* fd = &m_imgui_main_window.Frames[i];
				info.image = fd->Backbuffer;
				check(vkCreateImageView(m_instance.m_logical_device, &info, m_instance.mp_allocator, &fd->BackbufferView));
			}
		}

		// Create Framebuffer
		{
			VkImageView attachment[1]{};
			VkFramebufferCreateInfo info{};
			info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			info.renderPass = m_imgui_main_window.RenderPass;
			info.attachmentCount = 1;
			info.pAttachments = attachment;
			info.width = m_imgui_main_window.Width;
			info.height = m_imgui_main_window.Height;
			info.layers = 1;
			for (uint32_t i = 0; i < m_imgui_main_window.ImageCount; i++)
			{
				ImGui_ImplVulkanH_Frame* fd = &m_imgui_main_window.Frames[i];
				attachment[0] = fd->BackbufferView;
				check(vkCreateFramebuffer(m_instance.m_logical_device, &info, m_instance.mp_allocator, &fd->Framebuffer));
			}
		}
	}
	void VulkanWrapper::createCommandBuffers()
	{
		IM_ASSERT(m_instance.m_physical_device != VK_NULL_HANDLE && m_instance.m_logical_device != VK_NULL_HANDLE);

		for (uint32_t i = 0; i < m_imgui_main_window.ImageCount; i++)
		{
			ImGui_ImplVulkanH_Frame* fd = &m_imgui_main_window.Frames[i];
			ImGui_ImplVulkanH_FrameSemaphores* fsd = &m_imgui_main_window.FrameSemaphores[i];
			{
				VkCommandPoolCreateInfo info{};
				info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
				info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
				info.queueFamilyIndex = m_instance.m_queue_family_idx;
				check(vkCreateCommandPool(m_instance.m_logical_device, &info, m_instance.mp_allocator, &fd->CommandPool));
			}
			{
				VkCommandBufferAllocateInfo info{};
				info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
				info.commandPool = fd->CommandPool;
				info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				info.commandBufferCount = 1;
				check(vkAllocateCommandBuffers(m_instance.m_logical_device, &info, &fd->CommandBuffer));
			}
			{
				VkFenceCreateInfo info{};
				info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
				info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
				check(vkCreateFence(m_instance.m_logical_device, &info, m_instance.mp_allocator, &fd->Fence));
			}
			{
				VkSemaphoreCreateInfo info{};
				info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
				check(vkCreateSemaphore(m_instance.m_logical_device, &info, m_instance.mp_allocator, &fsd->ImageAcquiredSemaphore));
				check(vkCreateSemaphore(m_instance.m_logical_device, &info, m_instance.mp_allocator, &fsd->RenderCompleteSemaphore));
			}
		}
	}

	void VulkanWrapper::frameRender()
	{
		VkSemaphore image_acquired_semaphore =
			m_imgui_main_window.FrameSemaphores[m_imgui_main_window.SemaphoreIndex].ImageAcquiredSemaphore;
		VkSemaphore render_complete_semaphore =
			m_imgui_main_window.FrameSemaphores[m_imgui_main_window.SemaphoreIndex].RenderCompleteSemaphore;

		auto err = vkAcquireNextImageKHR(
			m_instance.m_logical_device,
			m_imgui_main_window.Swapchain,
			UINT64_MAX,
			image_acquired_semaphore,
			VK_NULL_HANDLE,
			&m_imgui_main_window.FrameIndex);
		if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
		{
			m_swapchain_rebuild = true;
			return;
		}
		check(err);

		ImGui_ImplVulkanH_Frame* frame = &m_imgui_main_window.Frames[m_imgui_main_window.FrameIndex];
		{
			check(vkWaitForFences(m_instance.m_logical_device, 1, &frame->Fence, VK_TRUE, UINT64_MAX));    // wait indefinitely instead of periodically checking
			check(vkResetFences(m_instance.m_logical_device, 1, &frame->Fence));
		}
		{
			check(vkResetCommandPool(m_instance.m_logical_device, frame->CommandPool, 0));
			VkCommandBufferBeginInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			check(vkBeginCommandBuffer(frame->CommandBuffer, &info));
		}
		{
			VkRenderPassBeginInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			info.renderPass = m_imgui_main_window.RenderPass;
			info.framebuffer = frame->Framebuffer;
			info.renderArea.extent.width = m_imgui_main_window.Width;
			info.renderArea.extent.height = m_imgui_main_window.Height;
			info.clearValueCount = 1;
			info.pClearValues = &m_imgui_main_window.ClearValue;
			vkCmdBeginRenderPass(frame->CommandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);
		}

		// Record dear imgui primitives into command buffer
		ImDrawData* draw_data = ImGui::GetDrawData();
		ImGui_ImplVulkan_RenderDrawData(draw_data, frame->CommandBuffer);

		// Submit command buffer
		vkCmdEndRenderPass(frame->CommandBuffer);
		{
			VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			VkSubmitInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			info.waitSemaphoreCount = 1;
			info.pWaitSemaphores = &image_acquired_semaphore;
			info.pWaitDstStageMask = &wait_stage;
			info.commandBufferCount = 1;
			info.pCommandBuffers = &frame->CommandBuffer;
			info.signalSemaphoreCount = 1;
			info.pSignalSemaphores = &render_complete_semaphore;

			check(vkEndCommandBuffer(frame->CommandBuffer));
			check(vkQueueSubmit(m_instance.m_queue, 1, &info, frame->Fence));
		}
	}
	void VulkanWrapper::framePresent()
	{
		if (m_swapchain_rebuild) return;

		VkSemaphore render_complete_semaphore =
			m_imgui_main_window.FrameSemaphores[m_imgui_main_window.SemaphoreIndex].RenderCompleteSemaphore;

		VkPresentInfoKHR info = {};
		info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		info.waitSemaphoreCount = 1;
		info.pWaitSemaphores = &render_complete_semaphore;
		info.swapchainCount = 1;
		info.pSwapchains = &m_imgui_main_window.Swapchain;
		info.pImageIndices = &m_imgui_main_window.FrameIndex;
		const VkResult err = vkQueuePresentKHR(m_instance.m_queue, &info);
		if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
		{
			m_swapchain_rebuild = true;
			return;
		}
		check(err);

		m_imgui_main_window.SemaphoreIndex =
			(m_imgui_main_window.SemaphoreIndex + 1) %
			m_imgui_main_window.ImageCount; // Now we can use the next set of semaphores
	}

	void VulkanWrapper::createImage(const Graphics::Bitmap& bitmap, VkCommandBuffer command_buffer)
	{
		const size_t image_byte_size = bitmap.GetWidth() * bitmap.GetHeight() * sizeof(bitmap.Value(0, 0));
		image_width = uint32_t(bitmap.GetWidth());
		image_height = uint32_t(bitmap.GetHeight());

		// Create the Image:
		{
			VkImageCreateInfo image_ci = {};
			image_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			image_ci.imageType = VK_IMAGE_TYPE_2D;
			image_ci.format = VK_FORMAT_R8G8B8A8_UNORM;
			image_ci.extent.width = uint32_t(bitmap.GetWidth());
			image_ci.extent.height = uint32_t(bitmap.GetWidth());
			image_ci.extent.depth = 1;
			image_ci.mipLevels = 1;
			image_ci.arrayLayers = 1;
			image_ci.samples = VK_SAMPLE_COUNT_1_BIT;
			image_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
			image_ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
			image_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			image_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			check(vkCreateImage(m_instance.m_logical_device, &image_ci, m_instance.mp_allocator, &m_image));

			VkMemoryRequirements req;
			vkGetImageMemoryRequirements(m_instance.m_logical_device, m_image, &req);

			VkMemoryAllocateInfo alloc_info = {};
			alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			alloc_info.allocationSize = req.size;
			alloc_info.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			check(vkAllocateMemory(m_instance.m_logical_device, &alloc_info, m_instance.mp_allocator, &m_device_memory));
			check(vkBindImageMemory(m_instance.m_logical_device, m_image, m_device_memory, 0));
		}

		// Create the Image View:
		{
			VkImageViewCreateInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			info.image = m_image;
			info.viewType = VK_IMAGE_VIEW_TYPE_2D;
			info.format = VK_FORMAT_R8G8B8A8_UNORM;
			info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			info.subresourceRange.levelCount = 1;
			info.subresourceRange.layerCount = 1;
			check(vkCreateImageView(m_instance.m_logical_device, &info, m_instance.mp_allocator, &m_image_view));
		}

		// image sampler
		{
			VkSamplerCreateInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
			info.magFilter = VK_FILTER_NEAREST;
			info.minFilter = VK_FILTER_NEAREST;
			info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
			info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
			info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
			info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
			info.minLod = -1000;
			info.maxLod = 1000;
			info.anisotropyEnable = VK_FALSE;
			info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_WHITE;
			check(vkCreateSampler(m_instance.m_logical_device, &info, m_instance.mp_allocator, &m_sampler));
		}


		VkDescriptorSet descriptor_set;
		{
			VkSampler sampler[1] = { m_sampler };
			VkDescriptorSetLayoutBinding binding[1]{};
			binding[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			binding[0].descriptorCount = 1;
			binding[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
			binding[0].pImmutableSamplers = sampler;

			VkDescriptorSetLayoutCreateInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			info.bindingCount = sizeof(binding) / sizeof(*binding);
			info.pBindings = binding;
			check(vkCreateDescriptorSetLayout(m_instance.m_logical_device, &info, m_instance.mp_allocator, &m_descriptor_set_layout));
		}
		{
			VkDescriptorSetAllocateInfo alloc_info = {};
			alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			alloc_info.descriptorPool = m_instance.m_descriptor_pool;
			alloc_info.descriptorSetCount = 1;
			alloc_info.pSetLayouts = &m_descriptor_set_layout;
			check(vkAllocateDescriptorSets(m_instance.m_logical_device, &alloc_info, &descriptor_set));
		}

		// Update the Descriptor Set:
		{
			VkDescriptorImageInfo desc_image[1] = {};
			desc_image[0].sampler = m_sampler;
			desc_image[0].imageView = m_image_view;
			desc_image[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			VkWriteDescriptorSet write_desc[1] = {};
			write_desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			write_desc[0].dstSet = descriptor_set;
			write_desc[0].descriptorCount = 1;
			write_desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			write_desc[0].pImageInfo = desc_image;
			vkUpdateDescriptorSets(m_instance.m_logical_device, 1, write_desc, 0, NULL);
		}
		m_render_texture = (ImTextureID)descriptor_set;

		// Create the Upload Buffer:
		{
			VkBufferCreateInfo buffer_info = {};
			buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			buffer_info.size = image_byte_size;
			buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
			buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			check(vkCreateBuffer(m_instance.m_logical_device, &buffer_info, m_instance.mp_allocator, &m_buffer));

			VkMemoryRequirements req;
			vkGetBufferMemoryRequirements(m_instance.m_logical_device, m_buffer, &req);

			VkMemoryAllocateInfo alloc_info = {};
			alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			alloc_info.allocationSize = req.size;
			alloc_info.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
			check(vkAllocateMemory(m_instance.m_logical_device, &alloc_info, m_instance.mp_allocator, &m_staging_memory));
			check(vkBindBufferMemory(m_instance.m_logical_device, m_buffer, m_staging_memory, 0));
		}

		// Upload to Buffer:
		{
			void* memory = nullptr;
			check(vkMapMemory(m_instance.m_logical_device, m_staging_memory, 0, image_byte_size, 0, &memory));
			std::memcpy(memory, bitmap.GetMapAddress(), image_byte_size);
			VkMappedMemoryRange range[1]{};
			range[0].sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
			range[0].memory = m_staging_memory;
			range[0].size = image_byte_size;
			check(vkFlushMappedMemoryRanges(m_instance.m_logical_device, sizeof(range) / sizeof(*range), range));
			vkUnmapMemory(m_instance.m_logical_device, m_staging_memory);
		}

		// Copy to Image:
		{
			VkImageMemoryBarrier copy_barrier[1]{};
			copy_barrier[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			copy_barrier[0].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			copy_barrier[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			copy_barrier[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			copy_barrier[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			copy_barrier[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			copy_barrier[0].image = m_image;
			copy_barrier[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			copy_barrier[0].subresourceRange.levelCount = 1;
			copy_barrier[0].subresourceRange.layerCount = 1;
			vkCmdPipelineBarrier(
				command_buffer,
				VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
				0, NULL,
				0, NULL,
				1, copy_barrier);

			VkBufferImageCopy region[1]{};
			region[0].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			region[0].imageSubresource.layerCount = 1;
			region[0].imageExtent.width = uint32_t(bitmap.GetWidth());
			region[0].imageExtent.height = uint32_t(bitmap.GetHeight());
			region[0].imageExtent.depth = 1;
			vkCmdCopyBufferToImage(
				command_buffer,
				m_buffer,
				m_image,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				uint32_t(sizeof(region) / sizeof(*region)), region);

			VkImageMemoryBarrier use_barrier[1]{};
			use_barrier[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			use_barrier[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			use_barrier[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			use_barrier[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			use_barrier[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			use_barrier[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			use_barrier[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			use_barrier[0].image = m_image;
			use_barrier[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			use_barrier[0].subresourceRange.levelCount = 1;
			use_barrier[0].subresourceRange.layerCount = 1;
			vkCmdPipelineBarrier(
				command_buffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
				0, NULL,
				0, NULL,
				1, use_barrier);
		}
	}
	void VulkanWrapper::updateImage(const Graphics::Bitmap& bitmap, VkCommandBuffer command_buffer)
	{
		if (bitmap.GetWidth() != image_width || bitmap.GetHeight() != image_height) return;

		const size_t image_byte_size = bitmap.GetWidth() * bitmap.GetHeight() * sizeof(bitmap.Value(0, 0));

		// Upload to Buffer:
		{
			void* memory = nullptr;
			check(vkMapMemory(m_instance.m_logical_device, m_staging_memory, 0, image_byte_size, 0, &memory));
			std::memcpy(memory, bitmap.GetMapAddress(), image_byte_size);
			VkMappedMemoryRange range[1]{};
			range[0].sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
			range[0].memory = m_staging_memory;
			range[0].size = image_byte_size;
			check(vkFlushMappedMemoryRanges(m_instance.m_logical_device, sizeof(range) / sizeof(*range), range));
			vkUnmapMemory(m_instance.m_logical_device, m_staging_memory);
		}

		// Copy to Image:
		{
			VkImageMemoryBarrier copy_barrier[1]{};
			copy_barrier[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			copy_barrier[0].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			copy_barrier[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			copy_barrier[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			copy_barrier[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			copy_barrier[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			copy_barrier[0].image = m_image;
			copy_barrier[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			copy_barrier[0].subresourceRange.levelCount = 1;
			copy_barrier[0].subresourceRange.layerCount = 1;
			vkCmdPipelineBarrier(
				command_buffer,
				VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
				0, NULL,
				0, NULL,
				1, copy_barrier);

			VkBufferImageCopy region[1]{};
			region[0].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			region[0].imageSubresource.layerCount = 1;
			region[0].imageExtent.width = uint32_t(bitmap.GetWidth());
			region[0].imageExtent.height = uint32_t(bitmap.GetHeight());
			region[0].imageExtent.depth = 1;
			vkCmdCopyBufferToImage(
				command_buffer,
				m_buffer,
				m_image,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				uint32_t(sizeof(region) / sizeof(*region)), region);

			VkImageMemoryBarrier use_barrier[1]{};
			use_barrier[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			use_barrier[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			use_barrier[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			use_barrier[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			use_barrier[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			use_barrier[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			use_barrier[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			use_barrier[0].image = m_image;
			use_barrier[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			use_barrier[0].subresourceRange.levelCount = 1;
			use_barrier[0].subresourceRange.layerCount = 1;
			vkCmdPipelineBarrier(
				command_buffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
				0, NULL,
				0, NULL,
				1, use_barrier);
		}
	}
	void VulkanWrapper::destroyImage()
	{
		if (m_buffer)
		{
			vkDestroyBuffer(m_instance.m_logical_device, m_buffer, m_instance.mp_allocator);
			m_buffer = VK_NULL_HANDLE;
		}
		if (m_staging_memory)
		{
			vkFreeMemory(m_instance.m_logical_device, m_staging_memory, m_instance.mp_allocator);
			m_staging_memory = VK_NULL_HANDLE;
		}

		if (m_image_view)
		{
			vkDestroyImageView(m_instance.m_logical_device, m_image_view, m_instance.mp_allocator);
			m_image_view = VK_NULL_HANDLE;
		}
		if (m_image)
		{
			vkDestroyImage(m_instance.m_logical_device, m_image, m_instance.mp_allocator);
			m_image = VK_NULL_HANDLE;
		}
		if (m_device_memory)
		{
			vkFreeMemory(m_instance.m_logical_device, m_device_memory, m_instance.mp_allocator);
			m_device_memory = VK_NULL_HANDLE;
		}
		if (m_sampler)
		{
			vkDestroySampler(m_instance.m_logical_device, m_sampler, m_instance.mp_allocator);
			m_sampler = VK_NULL_HANDLE;
		}
		if (m_descriptor_set_layout)
		{
			vkDestroyDescriptorSetLayout(m_instance.m_logical_device, m_descriptor_set_layout, m_instance.mp_allocator);
			m_descriptor_set_layout = VK_NULL_HANDLE;
		}
	}

	VkSurfaceFormatKHR VulkanWrapper::selectSurfaceFormat()
	{
		uint32_t available_count = 0;
		check(vkGetPhysicalDeviceSurfaceFormatsKHR(
			m_instance.m_physical_device, m_imgui_main_window.Surface,
			&available_count, NULL));
		std::vector<VkSurfaceFormatKHR> available_formats(available_count);
		vkGetPhysicalDeviceSurfaceFormatsKHR(
			m_instance.m_physical_device, m_imgui_main_window.Surface,
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
	VkPresentModeKHR VulkanWrapper::selectPresentMode()
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
		check(vkGetPhysicalDeviceSurfacePresentModesKHR(
			m_instance.m_physical_device, m_imgui_main_window.Surface,
			&available_count, NULL));
		std::vector<VkPresentModeKHR> available_modes(available_count);
		vkGetPhysicalDeviceSurfacePresentModesKHR(
			m_instance.m_physical_device, m_imgui_main_window.Surface,
			&available_count, available_modes.data());

		for (const auto& requested_mode : requested_modes)
			for (const auto& available_mode : available_modes)
				if (requested_mode == available_mode)
					return requested_mode;

		return VK_PRESENT_MODE_FIFO_KHR; // mandatory
	}

	void VulkanWrapper::createBuffer(
		VkDeviceSize image_byte_size,
		VkBufferUsageFlags usage_flags,
		VkMemoryPropertyFlags properties,
		VkBuffer& buffer, VkDeviceMemory& buffer_memory)
	{
		VkBufferCreateInfo buffer_ci{};
		buffer_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		buffer_ci.size = image_byte_size;
		buffer_ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		buffer_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		check(vkCreateBuffer(m_instance.m_logical_device, &buffer_ci, m_instance.mp_allocator, &buffer));

		VkMemoryRequirements memory_requirements;
		vkGetBufferMemoryRequirements(m_instance.m_logical_device, buffer, &memory_requirements);

		VkMemoryAllocateInfo allocate_info{};
		allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocate_info.allocationSize = memory_requirements.size;
		allocate_info.memoryTypeIndex = findMemoryType(memory_requirements.memoryTypeBits, properties);
		check(vkAllocateMemory(m_instance.m_logical_device, &allocate_info, m_instance.mp_allocator, &buffer_memory));

		check(vkBindBufferMemory(m_instance.m_logical_device, buffer, buffer_memory, 0));
	}
	uint32_t VulkanWrapper::findMemoryType(const uint32_t type_filter, const VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties memory_properties;
		vkGetPhysicalDeviceMemoryProperties(m_instance.m_physical_device, &memory_properties);

		for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
			if ((type_filter & (1 << i)) &&
				(memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		RZThrow("failed to find suitable mmemory type");
	}
}
