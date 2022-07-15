#pragma once

#include "rayzath.h"

#include "explorer_base.hpp"
#include "scene.hpp"
#include "properties.hpp"
#include "viewport.hpp"

#include <unordered_map>

namespace RZ = RayZath::Engine;

namespace RayZath::UI::Windows
{
	template <Engine::World::ObjectType T>
	class Explorer;

	using ObjectType = Engine::World::ObjectType;

	template<>
	class Explorer<ObjectType::Camera> : private ExplorerEditable
	{
	private:
		std::reference_wrapper<MultiProperties> mr_properties;
		std::reference_wrapper<Viewports> mr_viewports;
		RZ::Handle<RZ::Camera> m_selected, m_edited;

	public:
		Explorer(
			std::reference_wrapper<MultiProperties> properties,
			std::reference_wrapper<Viewports> viewports);

		void select(RZ::Handle<RZ::Camera> selected);
		void update(RZ::World& world);
	};
	template<>
	class Explorer<ObjectType::SpotLight> : private ExplorerEditable
	{
	private:
		std::reference_wrapper<MultiProperties> mr_properties;
		RZ::Handle<RZ::SpotLight> m_selected, m_edited;

	public:
		Explorer(std::reference_wrapper<MultiProperties> properties);

		void select(RZ::Handle<RZ::SpotLight> selected);
		void update(RZ::World& world);
	};
	template<>
	class Explorer<ObjectType::DirectLight> : private ExplorerEditable
	{
	private:
		std::reference_wrapper<MultiProperties> mr_properties;
		RZ::Handle<RZ::DirectLight> m_selected, m_edited;

	public:
		Explorer(std::reference_wrapper<MultiProperties> properties);

		void select(RZ::Handle<RZ::DirectLight> selected);
		void update(RZ::World& world);
	};
	
	template<>
	class Explorer<ObjectType::Material> : private ExplorerEditable
	{
	private:
		std::reference_wrapper<MultiProperties> mr_properties;
		RZ::Handle<RZ::Material> m_selected, m_edited;
		RZ::Material* mp_world_material = nullptr;

	public:
		Explorer(std::reference_wrapper<MultiProperties> properties);

		void select(RZ::Handle<RZ::Material> selected);
		void update(RZ::World& world);
	};
	
	template<>
	class Explorer<ObjectType::MeshStructure> : private ExplorerEditable
	{
	private:
		std::reference_wrapper<MultiProperties> mr_properties;
		RZ::Handle<RZ::MeshStructure> m_selected, m_edited;

	public:
		Explorer(std::reference_wrapper<MultiProperties> properties);

		void select(RZ::Handle<RZ::MeshStructure> selected);
		void update(RZ::World& world);
	};
	template<>
	class Explorer<ObjectType::Mesh> : private ExplorerEditable
	{
	private:
		std::reference_wrapper<MultiProperties> mr_properties;
		RZ::Handle<RZ::Mesh> m_selected_object, m_edited_object;
		RZ::Handle<RZ::Group> m_selected_group, m_edited_group;
		std::unordered_map<uint32_t, bool> m_object_ids, m_group_ids; // alraedy drawn objects

	public:
		Explorer(std::reference_wrapper<MultiProperties> properties);

		void select(RZ::Handle<RZ::Mesh> selected);
		void update(RZ::World& world);
	private:
		void renderTree(const RZ::Handle<RZ::Group>& group, RZ::World& world);
		void renderObject(const RZ::Handle<RZ::Mesh>& object, RZ::World& world);
	};

	class SceneExplorer : private ExplorerEditable
	{
	private:
		Scene& mr_scene;
		MultiProperties m_properties;

		std::reference_wrapper<Viewports> m_viewports;

		std::tuple<
			Explorer<ObjectType::Camera>,
			Explorer<ObjectType::SpotLight>,
			Explorer<ObjectType::DirectLight>,

			Explorer<ObjectType::Material>,

			Explorer<ObjectType::MeshStructure>,
			Explorer<ObjectType::Mesh>> m_explorers;
		bool m_selected = false;
		ObjectType m_selected_type{};

	public:
		SceneExplorer(Scene& scene, Viewports& viewports);

		void update();
		template <ObjectType T>
		void selectObject(const RZ::Handle<Engine::World::object_t<T>>& object)
		{
			std::get<Explorer<T>>(m_explorers).select(object);
			m_selected = true;
			m_selected_type = T;
		}
	};
}