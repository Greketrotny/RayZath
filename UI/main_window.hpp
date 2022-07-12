#pragma once

#include "scene.hpp"
#include "new_modals.hpp"

#include "imgui.h"

namespace RayZath::UI::Windows
{
	class Main
	{
	private:
		Scene& mr_scene;
		NewModals m_new_modals;

	public:
		Main(Scene& scene);

		void update();
	private:
		template <Engine::Material::Common T>
		void materialItem();
	};
}
