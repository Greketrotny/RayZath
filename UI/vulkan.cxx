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
		, m_window(m_instance)
	{}
	VulkanWrapper::~VulkanWrapper()
	{
		destroy();
	}
	
	void VulkanWrapper::init()
	{
		m_instance.init(m_glfw);
		m_window.init(m_glfw, m_glfw.frameBufferSize());
	}
	void VulkanWrapper::destroy()
	{
		//destroyImage();
		// ------------------
	}
	
	void VulkanWrapper::frameRender()
	{
		VkSemaphore image_acquired_semaphore = m_window.frame(m_window.semaphoreIndex()).m_semaphore.image_acquired;
		VkSemaphore render_complete_semaphore = m_window.frame(m_window.semaphoreIndex()).m_semaphore.render_complete;

		m_swapchain_rebuild = m_window.acquireNextImage();
		
		m_window.waitForFence();
		m_window.currentFrame().resetCommandPool();
		{
			VkClearColorValue clearColorValue;
			clearColorValue.float32[0] = 0.2f;
			clearColorValue.float32[1] = 1.0f;
			clearColorValue.float32[2] = 0.2f;
			clearColorValue.float32[3] = 0.5f;
			VkClearValue clearValue;
			clearValue.color = clearColorValue;
			VkRenderPassBeginInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			info.renderPass = m_window.renderPass();
			info.framebuffer = m_window.currentFrame().m_frame_buffer;
			info.renderArea.extent.width = m_window.resolution().x;
			info.renderArea.extent.height = m_window.resolution().y;
			info.clearValueCount = 1;
			info.pClearValues = &clearValue;
			vkCmdBeginRenderPass(
				m_window.currentFrame().m_command_buffer, 
				&info, 
				VK_SUBPASS_CONTENTS_INLINE);
		}

		// Record dear imgui primitives into command buffer
		ImDrawData* draw_data = ImGui::GetDrawData();
		ImGui_ImplVulkan_RenderDrawData(draw_data, m_window.currentFrame().m_command_buffer);

		// Submit command buffer
		vkCmdEndRenderPass(m_window.currentFrame().m_command_buffer);
		{
			VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			VkSubmitInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			info.waitSemaphoreCount = 1;
			info.pWaitSemaphores = &image_acquired_semaphore;
			info.pWaitDstStageMask = &wait_stage;
			info.commandBufferCount = 1;
			info.pCommandBuffers = &m_window.currentFrame().m_command_buffer;
			info.signalSemaphoreCount = 1;
			info.pSignalSemaphores = &render_complete_semaphore;

			check(vkEndCommandBuffer(m_window.currentFrame().m_command_buffer));
			check(vkQueueSubmit(m_instance.queue(), 1, &info, m_window.currentFrame().m_fence));
		}
	}
	void VulkanWrapper::framePresent()
	{
		if (m_swapchain_rebuild) return;

		VkSemaphore render_complete_semaphore = m_window.frame(m_window.semaphoreIndex()).m_semaphore.render_complete;

		auto image_index = m_window.frameIndex();
		auto swapchain = m_window.swapchain();
		VkPresentInfoKHR info{};
		info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		info.waitSemaphoreCount = 1;
		info.pWaitSemaphores = &render_complete_semaphore;
		info.swapchainCount = 1;
		info.pSwapchains = &swapchain;
		info.pImageIndices = &image_index;
		const VkResult err = vkQueuePresentKHR(m_instance.queue(), &info);
		if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
		{
			m_swapchain_rebuild = true;
			return;
		}
		check(err);

		m_window.incrementSemaphoreIndex();
	}

	/*void VulkanWrapper::createImage(const Graphics::Bitmap& bitmap, VkCommandBuffer command_buffer)
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
	}*/
}
