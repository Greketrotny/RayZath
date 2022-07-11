#pragma once

#include "scene.hpp"

#include "imgui.h"

namespace RayZath::UI::Windows
{
	class Main
	{
	private:
		Scene& mr_scene;

	public:
		Main(Scene& scene);

		void update();
	private:
		template <Engine::Material::Common T>
		void materialItem();
	};
}
