module;

#include "rayzath.h"
#include "imgui.h"

export module rz.ui.windows.viewport;

import rz.ui.rendering.vulkan.image;

namespace RZ = RayZath::Engine;

export namespace RayZath::UI::Windows
{
	class Viewport
	{
	private:
		RZ::Handle<RZ::Camera> m_camera;

		Math::vec2i32 m_mouse_click_position, m_mouse_previous_position;
		Math::vec2f32 m_mouse_click_rotation;
		Math::vec3f m_polar_rotation_origin, m_mouse_click_polar_rotation;

		bool was_focused = false;
		Math::vec2u32 m_previous_resolution;

		float m_zoom = 1.0f;
		bool m_fit_to_viewport = false;
		bool m_auto_fit = true;
		Math::vec2f32 m_old_image_pos{}, m_image_pos{}, m_click_pos{};

	public:
		void setCamera(RZ::Handle<RZ::Camera> camera);

		void update(const float dt, const Rendering::Vulkan::Image& image);
	private:
		void drawMenu();
		void controlCamera(const float dt);
		void drawRender(const Rendering::Vulkan::Image& image);
	};
}
