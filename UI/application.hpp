#pragma once

#include "rayzath.h"

#include "rendering.hpp"

#include "main_window.hpp"

#include "scene.hpp"

namespace RayZath::UI
{
	class Application
	{
	public:
		Rendering::Module m_rendering;
		Scene m_scene;
		Windows::Main m_main_window;

	public:
		Application();

		static Application& instance();
		int run();

		void update();
		void render();
	};
}
