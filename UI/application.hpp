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

		Windows::Viewports m_viewports;
		Windows::Main m_main;
		Windows::Explorer m_explorer;

		Scene m_scene;

	public:
		Application();

		static Application& instance();
		int run();

		void update();
		void render();
	};
}
