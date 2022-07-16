#pragma once

#include "scene.hpp"
#include "new_modals.hpp"
#include "load_modals.hpp"

#include "explorer.hpp" // MapObjectType

namespace RayZath::UI::Windows
{
	class SceneExplorer; 

	class Main
	{
	private:
		Scene& mr_scene;
		NewModals m_new_modals;
		LoadModals m_load_modals;

	public:
		Main(Scene& scene);

		void update(SceneExplorer&);
	private:
		template <Engine::Material::Common T>
		void materialItem(SceneExplorer&);
		template <Engine::World::ObjectType T> requires MapObjectType<T>
		void mapItem(SceneExplorer&);
	};
}
