#include "image.hpp"

#include "rzexception.hpp"

namespace RayZath::UI::Rendering::Vulkan
{
	Image::~Image()
	{
		destroyImage();
	}

	void Image::createImage(const Graphics::Bitmap& bitmap, VkCommandBuffer command_buffer)
	{
		const std::size_t image_byte_size = bitmap.GetWidth() * bitmap.GetHeight() * sizeof(bitmap.Value(0, 0));
		image_width = std::max(uint32_t(1), uint32_t(bitmap.GetWidth()));
		image_height = std::max(uint32_t(1), uint32_t(bitmap.GetHeight()));

		// Create the Image:
		{
			VkImageCreateInfo image_ci = {};
			image_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			image_ci.imageType = VK_IMAGE_TYPE_2D;
			image_ci.format = VK_FORMAT_R8G8B8A8_UNORM;
			image_ci.extent.width = uint32_t(bitmap.GetWidth());
			image_ci.extent.height = uint32_t(bitmap.GetHeight());
			image_ci.extent.depth = 1;
			image_ci.mipLevels = 1;
			image_ci.arrayLayers = 1;
			image_ci.samples = VK_SAMPLE_COUNT_1_BIT;
			image_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
			image_ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
			image_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			image_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			check(vkCreateImage(Instance::get().logicalDevice(), &image_ci, Instance::get().allocator(), &m_image));

			VkMemoryRequirements req;
			vkGetImageMemoryRequirements(Instance::get().logicalDevice(), m_image, &req);

			VkMemoryAllocateInfo alloc_info = {};
			alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			alloc_info.allocationSize = req.size;
			alloc_info.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			check(vkAllocateMemory(Instance::get().logicalDevice(), &alloc_info, Instance::get().allocator(), &m_device_memory));
			check(vkBindImageMemory(Instance::get().logicalDevice(), m_image, m_device_memory, 0));
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
			check(vkCreateImageView(Instance::get().logicalDevice(), &info, Instance::get().allocator(), &m_image_view));
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
			check(vkCreateSampler(Instance::get().logicalDevice(), &info, Instance::get().allocator(), &m_sampler));
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
			check(vkCreateDescriptorSetLayout(Instance::get().logicalDevice(), &info, Instance::get().allocator(), &m_descriptor_set_layout));
		}
		{
			VkDescriptorSetAllocateInfo alloc_info = {};
			alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			alloc_info.descriptorPool = Instance::get().descriptorPool();
			alloc_info.descriptorSetCount = 1;
			alloc_info.pSetLayouts = &m_descriptor_set_layout;
			check(vkAllocateDescriptorSets(Instance::get().logicalDevice(), &alloc_info, &descriptor_set));
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
			vkUpdateDescriptorSets(Instance::get().logicalDevice(), 1, write_desc, 0, NULL);
		}
		m_texture_id = (ImTextureID)descriptor_set;

		// Create the Upload Buffer:
		{
			VkBufferCreateInfo buffer_info = {};
			buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			buffer_info.size = image_byte_size;
			buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
			buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			check(vkCreateBuffer(Instance::get().logicalDevice(), &buffer_info, Instance::get().allocator(), &m_buffer));

			VkMemoryRequirements req;
			vkGetBufferMemoryRequirements(Instance::get().logicalDevice(), m_buffer, &req);

			VkMemoryAllocateInfo alloc_info = {};
			alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			alloc_info.allocationSize = m_staging_memory_size = req.size;
			alloc_info.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
			check(vkAllocateMemory(Instance::get().logicalDevice(), &alloc_info, Instance::get().allocator(), &m_staging_memory));
			check(vkBindBufferMemory(Instance::get().logicalDevice(), m_buffer, m_staging_memory, 0));
		}

		uploadImage(bitmap, command_buffer);
	}
	void Image::updateImage(const Graphics::Bitmap& bitmap, VkCommandBuffer command_buffer)
	{
		if (bitmap.GetWidth() == 0 || bitmap.GetHeight() == 0) return;
		if (bitmap.GetWidth() != image_width || bitmap.GetHeight() != image_height)
		{
			destroyImage();
			createImage(bitmap, command_buffer);
			return;
		}

		uploadImage(bitmap, command_buffer);	
	}
	void Image::destroyImage()
	{
		if (m_buffer)
		{
			vkDestroyBuffer(Instance::get().logicalDevice(), m_buffer, Instance::get().allocator());
			m_buffer = VK_NULL_HANDLE;
		}
		if (m_staging_memory)
		{
			vkFreeMemory(Instance::get().logicalDevice(), m_staging_memory, Instance::get().allocator());
			m_staging_memory = VK_NULL_HANDLE;
		}

		if (m_image_view)
		{
			vkDestroyImageView(Instance::get().logicalDevice(), m_image_view, Instance::get().allocator());
			m_image_view = VK_NULL_HANDLE;
		}
		if (m_image)
		{
			vkDestroyImage(Instance::get().logicalDevice(), m_image, Instance::get().allocator());
			m_image = VK_NULL_HANDLE;
		}
		if (m_device_memory)
		{
			vkFreeMemory(Instance::get().logicalDevice(), m_device_memory, Instance::get().allocator());
			m_device_memory = VK_NULL_HANDLE;
		}
		if (m_sampler)
		{
			vkDestroySampler(Instance::get().logicalDevice(), m_sampler, Instance::get().allocator());
			m_sampler = VK_NULL_HANDLE;
		}
		if (m_descriptor_set_layout)
		{
			vkDestroyDescriptorSetLayout(Instance::get().logicalDevice(), m_descriptor_set_layout, Instance::get().allocator());
			m_descriptor_set_layout = VK_NULL_HANDLE;
			m_texture_id = nullptr;
		}
	}
	void Image::uploadImage(const Graphics::Bitmap& bitmap, VkCommandBuffer command_buffer)
	{
		const std::size_t image_byte_size = bitmap.GetWidth() * bitmap.GetHeight() * sizeof(bitmap.Value(0, 0));

		// copy data to buffer
		{
			void* memory = nullptr;
			check(vkMapMemory(Instance::get().logicalDevice(), m_staging_memory, 0, m_staging_memory_size, 0, &memory));
			std::memcpy(memory, bitmap.GetMapAddress(), image_byte_size);
			VkMappedMemoryRange range[1]{};
			range[0].sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
			range[0].memory = m_staging_memory;
			range[0].size = m_staging_memory_size;
			check(vkFlushMappedMemoryRanges(Instance::get().logicalDevice(), sizeof(range) / sizeof(*range), range));
			vkUnmapMemory(Instance::get().logicalDevice(), m_staging_memory);
		}

		// append copy commands and memory barriers
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

	void Image::createBuffer(
		VkDeviceSize image_byte_size,
		[[maybe_unused]] VkBufferUsageFlags usage_flags,
		VkMemoryPropertyFlags properties,
		VkBuffer& buffer, VkDeviceMemory& buffer_memory)
	{
		VkBufferCreateInfo buffer_ci{};
		buffer_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		buffer_ci.size = image_byte_size;
		buffer_ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		buffer_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		check(vkCreateBuffer(Instance::get().logicalDevice(), &buffer_ci, Instance::get().allocator(), &buffer));

		VkMemoryRequirements memory_requirements;
		vkGetBufferMemoryRequirements(Instance::get().logicalDevice(), buffer, &memory_requirements);

		VkMemoryAllocateInfo allocate_info{};
		allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocate_info.allocationSize = memory_requirements.size;
		allocate_info.memoryTypeIndex = findMemoryType(memory_requirements.memoryTypeBits, properties);
		check(vkAllocateMemory(Instance::get().logicalDevice(), &allocate_info, Instance::get().allocator(), &buffer_memory));

		check(vkBindBufferMemory(Instance::get().logicalDevice(), buffer, buffer_memory, 0));
	}

	uint32_t Image::findMemoryType(const uint32_t type_filter, const VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties memory_properties;
		vkGetPhysicalDeviceMemoryProperties(Instance::get().physicalDevice(), &memory_properties);

		for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
			if ((type_filter & (1 << i)) &&
				(memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		RZThrow("failed to find suitable mmemory type");
	}
}
