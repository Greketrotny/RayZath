#pragma once

#include "scene.hpp"
#include "explorer_base.hpp"

#include <variant>

namespace RayZath::UI::Windows
{
	class SaveModalBase
	{
	protected:
		bool m_opened = true;
		std::array<char, 2048> m_path_buffer{};
		std::optional<std::string> m_fail_message;
	};
	template <Engine::World::ObjectType T>
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
		std::unique_ptr<Search<Engine::World::ObjectType::Material>> m_search_modal;
		Engine::Handle<Engine::Material> m_selected_material;

	public:
		void update(Scene& scene);
	};
	class OBJSaveModal : public SaveModalBase
	{
	private:
		std::unique_ptr<Search<Engine::World::ObjectType::MeshStructure>> m_search_modal;
		Engine::Handle<Engine::MeshStructure> m_selected_mesh;

	public:
		void update(Scene& scene);
	};
	class InstanceSaveModal : public SaveModalBase
	{
	private:
		std::unique_ptr<Search<Engine::World::ObjectType::Mesh>> m_search_modal;
		Engine::Handle<Engine::Mesh> m_selected_instance;

	public:
		void update(Scene& scene);
	};
	class ModelSaveModal : public SaveModalBase
	{
	private:
		std::unique_ptr<Search<Engine::World::ObjectType::Group>> m_search_modal;
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
			MapSaveModal<Engine::World::ObjectType::Texture>,
			MapSaveModal<Engine::World::ObjectType::NormalMap>,
			MapSaveModal<Engine::World::ObjectType::MetalnessMap>,
			MapSaveModal<Engine::World::ObjectType::RoughnessMap>,
			MapSaveModal<Engine::World::ObjectType::EmissionMap>,
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
