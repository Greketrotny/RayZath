#pragma once

#include "scene.hpp"

namespace RayZath::UI::Windows
{
	class Main
	{
	private:
		Scene& mr_scene;

	public:
		Main(Scene& scene);

		void update();
	};
}
