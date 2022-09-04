#pragma once

#include "rayzath.hpp"

#include "image.hpp"

namespace RayZath::UI::Windows
{
	namespace RZ = RayZath::Engine;

	struct Stats
	{
		RZ::Timer timer;
		uint64_t prev_ray_count = 0;
		float ft = 16.666f;
	};

	class Viewport
	{
	private:
		std::reference_wrapper<RZ::World> mr_world;
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

		// save render image
		std::array<char, 1024> m_path_buffer{};
		std::string m_error_message;

		// stats
		std::unique_ptr<Stats> m_stats;

	public:
		Viewport(std::reference_wrapper<RZ::World> world, RZ::Handle<RZ::Camera> camera, const uint32_t id);

		const auto& camera() const { return m_camera; }

		void update(const Rendering::Vulkan::Handle<VkCommandBuffer>& command_buffer);
		void draw();
		bool valid() const;
		bool clicked();
	private:
		void drawMenu();
		void controlCamera();
		void drawRender();
		void drawStats();
	};

	class Viewports
	{
	private:
		std::reference_wrapper<RZ::World> mr_world;
		std::unordered_map<uint32_t, Viewport> m_viewports;

	public:
		Viewports(std::reference_wrapper<RZ::World> world)
			: mr_world(world)
		{}

		Viewport& addViewport(RZ::Handle<RZ::Camera> camera);
		void destroyInvalidViewports();

		void update(const Rendering::Vulkan::Handle<VkCommandBuffer>& command_buffer);
		void draw();
		RZ::Handle<RZ::Camera> getSelected();
	};
}
