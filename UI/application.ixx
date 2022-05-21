module;

#include "rayzath.h"

export module rz.ui.application;

export import rz.ui.rendering;
export import rz.ui.windows.viewport;

export namespace RayZath::UI
{
	class Application
	{
	public:
		Rendering m_rendering;
		Viewport m_viewport;

	public:
		static Application& instance();
		int run();

		void draw(Graphics::Bitmap&& bitmap);
	};
}
