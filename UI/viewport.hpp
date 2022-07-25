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
		uint32_t m_id;

		Math::vec2i32 m_mouse_click_position, m_mouse_previous_position;
		Math::vec2f32 m_mouse_click_rotation;
		Math::vec3f m_polar_rotation_origin, m_mouse_click_polar_rotation;

		bool was_focused = false;
		bool m_clicked = false;
		bool is_opened = true;
		Math::vec2u32 m_previous_resolution;

		// canvas
		float m_zoom = 1.0f;
		bool m_fit_to_viewport = false;
		bool m_auto_fit = false;
		Math::vec2f32 m_old_image_pos{}, m_image_pos{}, m_click_pos{};

		// animation
		float m_rotation_angle = 0.0f;
		float m_rotation_speed = 0.05f;
		Math::vec3f32 m_rotation_center{}, m_rotation_vector{};
		bool m_animate = false;

	public:
		Viewport(RZ::Handle<RZ::Camera> camera, const uint32_t id);

		const auto& camera() const { return m_camera; }

		void update(const Rendering::Vulkan::Handle<VkCommandBuffer>& command_buffer);
		void draw();
		bool valid() const;
		bool clicked();
	private:
		void drawMenu();
		void controlCamera();
		void drawRender();
	};

	class Viewports
	{
	private:
		std::unordered_map<uint32_t, Viewport> m_viewports;

	public:
		Viewport& addViewport(RZ::Handle<RZ::Camera> camera);
		void destroyInvalidViewports();

		void update(const Rendering::Vulkan::Handle<VkCommandBuffer>& command_buffer);
		void draw();
		RZ::Handle<RZ::Camera> getSelected();
	};
}
