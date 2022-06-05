module;

#include "rayzath.h"

export module rz.ui.application;

export import rz.ui.rendering;
export import rz.ui.windows.viewport;
export import rz.ui.windows.explorer;

export import rz.ui.scene;

export namespace RayZath::UI
{
	class Application
	{
	public:
		Rendering::RenderingWrapper m_rendering;

		Viewport m_viewport;
		Windows::Explorer m_explorer;


		Scene m_scene;

	public:
		static Application& instance();
		int run();

		void update();
	};
}
