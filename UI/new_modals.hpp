#pragma once

#include "scene.hpp"

#include <variant>

namespace RayZath::UI::Windows
{
	class SceneExplorer;

	template <CommonMesh T>
	class NewMeshModal
	{
	protected:
		bool m_opened = true;
		CommonMeshParameters<T> m_parameters;
		std::reference_wrapper<SceneExplorer> mr_explorer;

	public:
		NewMeshModal(std::reference_wrapper<SceneExplorer> explorer)
			: mr_explorer(std::move(explorer))
		{}
		void update(Scene& scene);
	};

	class NewMaterialModal
	{
	protected:
		bool m_opened = true;
		Engine::ConStruct<Engine::Material> m_construct;
		std::reference_wrapper<SceneExplorer> mr_explorer;

	public:
		NewMaterialModal(std::reference_wrapper<SceneExplorer> explorer)
			: mr_explorer(std::move(explorer))
		{}

		void update(Scene& scene);
	};

	class NewModals
	{
	private:
		std::reference_wrapper<Scene> mr_scene;
		std::variant<
			std::monostate, 
			NewMeshModal<CommonMesh::Plane>,
			NewMeshModal<CommonMesh::Sphere>,
			NewMeshModal<CommonMesh::Cone>,
			NewMeshModal<CommonMesh::Cylinder>,
			NewMeshModal<CommonMesh::Torus>,		
			NewMaterialModal> m_new_modal;

	public:
		NewModals(std::reference_wrapper<Scene> scene)
			: mr_scene(std::move(scene))
		{}

		template <typename T>
		void open(std::reference_wrapper<SceneExplorer> explorer)
		{
			m_new_modal.emplace<T>(std::move(explorer));
		}
		void update()
		{
			std::visit([ref = mr_scene](auto& modal)
				{
					if constexpr (!std::is_same_v<std::decay_t<decltype(modal)>, std::monostate>)
						modal.update(ref);
				}, m_new_modal);
		}
	};
}
