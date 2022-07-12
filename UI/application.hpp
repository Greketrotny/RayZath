#pragma once

#include "rayzath.h"

#include "rendering.hpp"

#include "main_window.hpp"
#include "explorer.hpp"
#include "viewport.hpp"

#include "scene.hpp"

namespace RayZath::UI
{
	class Application
	{
	public:
		Rendering::Module m_rendering;
		Scene m_scene;

		Windows::Viewports m_viewports;
		Windows::Main m_main;
		Windows::SceneExplorer m_explorer;


	public:
		Application();

		static Application& instance();
		int run();

		void update();
		void render();
	};
}
