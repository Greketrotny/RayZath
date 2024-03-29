#pragma once

#include "vulkan/vulkan.h"
#include "imgui.h"

#include "instance.hpp"

#include "bitmap.h"

namespace RayZath::UI::Rendering::Vulkan
{
	class Image
	{
	private:
		uint32_t image_width{}, image_height{};
		Handle<VkImage> m_image{};
		Handle<VkImageView> m_image_view{};
		Handle<VkSampler> m_sampler{};
		Handle<VkDescriptorSetLayout> m_descriptor_set_layout{};
		Handle<VkDeviceMemory> m_device_memory{};
		ImTextureID m_texture_id = nullptr;

		Handle<VkBuffer> m_buffer{};
		Handle<VkDeviceMemory> m_staging_memory{};
		uint64_t m_staging_memory_size{};

	public:
		Image(const Image&) = delete;
		Image(Image&&) = default;
		Image() = default;
		~Image();

		Image& operator=(const Image&) = delete;
		Image& operator=(Image&&) = default;

		auto width() const { return image_width; }
		auto height() const { return image_height; }
		auto textureHandle() const { return m_texture_id; }

		void createImage(const Graphics::Bitmap& bitmap, VkCommandBuffer command_buffer);
		void updateImage(const Graphics::Bitmap& bitmap, VkCommandBuffer command_buffer);
	private:
		void destroyImage();
		void uploadImage(const Graphics::Bitmap& bitmap, VkCommandBuffer command_buffer);

	private:
		void createBuffer(
			VkDeviceSize size,
			VkBufferUsageFlags usage_flags,
			VkMemoryPropertyFlags properties,
			VkBuffer& buffer, VkDeviceMemory& buffer_memory);
		uint32_t findMemoryType(const uint32_t type_filter, const VkMemoryPropertyFlags properties);
	};
}
