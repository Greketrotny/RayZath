module;

#include "rayzath.h"
#include "imgui.h"

export module rz.ui.windows.viewport;

import rz.ui.rendering.vulkan.image;

namespace RZ = RayZath::Engine;

export namespace RayZath::UI
{
	class Viewport
	{
	private:
		RZ::Handle<RZ::Camera> m_camera;

		Math::vec2i32 m_mouse_click_position, m_mouse_previous_position;
		Math::vec2f m_mouse_click_rotation;
		bool was_focused = false;

		Math::vec3f m_polar_rotation_origin, m_mouse_click_polar_rotation;

	public:
		void setCamera(RZ::Handle<RZ::Camera> camera);

		void update(const float dt, const Rendering::Vulkan::Image& image);
	};
}
