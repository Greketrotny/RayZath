#pragma once

#include "scene.hpp"
#include "new_modals.hpp"

#include "imgui.h"

namespace RayZath::UI::Windows
{
	class SceneExplorer; 

	class Main
	{
	private:
		Scene& mr_scene;
		NewModals m_new_modals;

	public:
		Main(Scene& scene);

		void update(SceneExplorer&);
	private:
		template <Engine::Material::Common T>
		void materialItem(SceneExplorer&);
	};
}
