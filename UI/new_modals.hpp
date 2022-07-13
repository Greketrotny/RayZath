#pragma once

#include "scene.hpp"

#include <variant>

namespace RayZath::UI::Windows
{
	template <CommonMesh T>
	class NewModal
	{
	protected:
		bool m_opened = true;
		CommonMeshParameters<T> m_parameters;

	public:
		void update(Scene& scene);
	};

	class NewModals
	{
	private:
		std::reference_wrapper<Scene> mr_scene;
		std::variant<
			std::monostate, 
			NewModal<CommonMesh::Plane>, 
			NewModal<CommonMesh::Sphere>,
			NewModal<CommonMesh::Cylinder>> m_new_modal;

	public:
		NewModals(std::reference_wrapper<Scene> scene)
			: mr_scene(std::move(scene))
		{}

		template <typename T>
		void open()
		{
			m_new_modal.emplace<T>();
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
