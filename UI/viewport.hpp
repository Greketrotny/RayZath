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

		// viewport
		bool m_was_focused = false;
		bool m_clicked = false;
		bool m_is_opened = true;
		bool m_resized = false;
		bool m_selected = false, m_requested_select = false;
		Math::vec2i32 m_content_min, m_content_max;
		Math::vec2i32 m_prev_content_res, m_content_res;

		Math::vec2i32 m_content_mouse_pos, m_content_mouse_click_pos, m_content_mouse_prev_pos;
		bool m_mouse_dragging = false;
		Math::vec2f32 m_mouse_click_rotation;
		Math::vec3f m_polar_rotation_origin, m_mouse_click_polar_rotation;

		// canvas
		float m_zoom = 1.0f;
		bool m_fit_to_viewport = false;
		bool m_auto_fit = false;
		bool m_reset_canvas = false;
		Math::vec2f32 m_canvas_center_pos, m_canvas_center_click_pos;
		Math::vec2i32 m_image_click_pos;
		Engine::Handle<Engine::Instance> m_selected_mesh;
		Engine::Handle<Engine::Material> m_selected_material;

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

		bool valid() const;
		bool clicked();
		bool selected();
		const auto& camera() const { return m_camera; }
		const auto& getSelectedMesh() const { return m_selected_mesh; }

		void draw(const Rendering::Vulkan::Handle<VkCommandBuffer>& command_buffer);
		void update(const Rendering::Vulkan::Handle<VkCommandBuffer>& command_buffer);
	private:
		void drawMenu();
		void controlCamera();
		void controlCanvas();
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

		void draw(const Rendering::Vulkan::Handle<VkCommandBuffer>& command_buffer);
		RZ::Handle<RZ::Camera> getSelected();
		RZ::Handle<RZ::Instance> getSelectedMesh();
		RZ::Handle<RZ::Material> getSelectedMaterial();
	};
}
