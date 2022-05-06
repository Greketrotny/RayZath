export module rz.ui.application;

import rz.ui.rendering;
import rz.ui.windows.viewport;

import "rayzath.h";

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
