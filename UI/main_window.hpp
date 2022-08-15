#pragma once

#include "explorer.hpp"
#include "viewport.hpp"
#include "settings.hpp"

#include "new_modals.hpp"
#include "load_modals.hpp"
#include "save_modals.hpp"

#include "scene.hpp"
#include "rendering.hpp"

namespace RayZath::UI::Windows
{
	class Main
	{
	private:
		Scene& mr_scene;
		Rendering::Module& mr_rendering;

		Windows::Viewports m_viewports;
		Windows::SceneExplorer m_explorer;
		Windows::Settings m_settings;

		NewModals m_new_modals;
		LoadModals m_load_modals;
		SaveModals m_save_modals;

	public:
		Main(Scene& scene, Rendering::Module& rendering);

		void update();
	private:
		template <Engine::Material::Common T>
		void materialItem();
		template <Engine::World::ObjectType T> requires MapObjectType<T>
		void mapItem();

		void updateMenuBar();
	};
}
