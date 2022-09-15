#pragma once

#include "explorer.hpp"
#include "viewport.hpp"
#include "settings.hpp"
#include "render_state.hpp"

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

		Viewports m_viewports;
		SceneExplorer m_explorer;
		Settings m_settings;
		RenderState m_render_state;

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
