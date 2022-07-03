#pragma once

#include "rayzath.h"

#include "image.hpp"

namespace RayZath::UI::Windows
{
	namespace RZ = RayZath::Engine;

	class Viewport
	{
	private:
		RZ::Handle<RZ::Camera> m_camera;
		Rendering::Vulkan::Image m_image;

		Math::vec2i32 m_mouse_click_position, m_mouse_previous_position;
		Math::vec2f32 m_mouse_click_rotation;
		Math::vec3f m_polar_rotation_origin, m_mouse_click_polar_rotation;

		bool was_focused = false;
		bool is_opened = true;
		Math::vec2u32 m_previous_resolution;

		float m_zoom = 1.0f;
		bool m_fit_to_viewport = false;
		bool m_auto_fit = true;
		Math::vec2f32 m_old_image_pos{}, m_image_pos{}, m_click_pos{};

	public:
		Viewport(RZ::Handle<RZ::Camera> camera);

		const auto& camera() const { return m_camera; }

		void update(const Rendering::Vulkan::Handle<VkCommandBuffer>& command_buffer);
		void draw();
		bool valid() const;
	private:
		void drawMenu();
		void controlCamera();
		void drawRender();
	};

	class Viewports
	{
	private:
		std::vector<Viewport> m_viewports;

	public:
		Viewport& addViewport(RZ::Handle<RZ::Camera> camera);
		void destroyInvalidViewports();

		void update(const Rendering::Vulkan::Handle<VkCommandBuffer>& command_buffer);
		void draw();
	};
}
