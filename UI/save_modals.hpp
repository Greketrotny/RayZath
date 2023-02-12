#pragma once

#include "scene.hpp"
#include "explorer_base.hpp"
#include "file_browser.hpp"
#include "message_box.hpp"

#include <variant>

namespace RayZath::UI::Windows
{
	class SaveModalBase
	{
	protected:
		bool m_opened = true;
		MessageBox m_message_box;
		std::filesystem::path m_file_to_save;
		std::optional<FileBrowserModal> m_file_browser;

		void updateFileBrowsing();
	};
	template <Engine::ObjectType T>
	class MapSaveModal : public SaveModalBase
	{
	private:
		std::unique_ptr<Search<T>> m_search_modal;
		Engine::Handle<Engine::World::object_t<T>> m_selected_map;

	public:
		void update(Scene& scene);
	};
	class MTLSaveModal : public SaveModalBase
	{
	private:
		std::unique_ptr<Search<Engine::ObjectType::Material>> m_search_modal;
		Engine::Handle<Engine::Material> m_selected_material;

	public:
		void update(Scene& scene);
	};
	class OBJSaveModal : public SaveModalBase
	{
	private:
		std::unique_ptr<Search<Engine::ObjectType::Mesh>> m_search_modal;
		Engine::Handle<Engine::Mesh> m_selected_mesh;

	public:
		void update(Scene& scene);
	};
	class InstanceSaveModal : public SaveModalBase
	{
	private:
		std::unique_ptr<Search<Engine::ObjectType::Instance>> m_search_modal;
		Engine::Handle<Engine::Instance> m_selected_instance;

	public:
		void update(Scene& scene);
	};
	class ModelSaveModal : public SaveModalBase
	{
	private:
		std::unique_ptr<Search<Engine::ObjectType::Group>> m_search_modal;
		Engine::Handle<Engine::Group> m_selected_group;

	public:
		void update(Scene& scene);
	};
	class SceneSaveModal : public SaveModalBase
	{
	private:
		Engine::Saver::SaveOptions m_save_options;

	public:
		void update(Scene& scene);
	};

	class SaveModals
	{
	private:
		std::reference_wrapper<Scene> mr_scene;
		std::variant<
			std::monostate,
			MapSaveModal<Engine::ObjectType::Texture>,
			MapSaveModal<Engine::ObjectType::NormalMap>,
			MapSaveModal<Engine::ObjectType::MetalnessMap>,
			MapSaveModal<Engine::ObjectType::RoughnessMap>,
			MapSaveModal<Engine::ObjectType::EmissionMap>,
			MTLSaveModal,
			OBJSaveModal,
			InstanceSaveModal,
			ModelSaveModal,
			SceneSaveModal
		> m_load_modal;

	public:
		SaveModals(std::reference_wrapper<Scene> scene)
			: mr_scene(std::move(scene))
		{}

		template <typename T>
		void open()
		{
			m_load_modal.emplace<T>();
		}
		void update()
		{
			std::visit([ref = mr_scene](auto& modal)
				{
					if constexpr (!std::is_same_v<std::decay_t<decltype(modal)>, std::monostate>)
					modal.update(ref);
				}, m_load_modal);
		}
	};
}
