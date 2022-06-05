module;

#include "rayzath.h"
#include "imgui.h"

export module rz.ui.windows.viewport;

import rz.ui.rendering.vulkan.image;

export namespace RayZath::UI
{
	class Viewport
	{
	private:
	public:
		Math::vec2f m_resolution;

		void draw(Rendering::Vulkan::Image& image);
	};
}
