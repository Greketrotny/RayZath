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

	public:
		void setCamera(RZ::Handle<RZ::Camera> camera);
		void draw(Rendering::Vulkan::Image& image);
	};
}
