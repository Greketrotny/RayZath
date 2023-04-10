#pragma once

#include "rayzath.hpp"

#include "explorer_base.hpp"
#include "scene.hpp"
#include "properties.hpp"
#include "viewport.hpp"

#include "dictionary.hpp"

#include <unordered_map>

namespace RZ = RayZath::Engine;

namespace RayZath::UI::Windows
{
	template <Engine::ObjectType T>
	class Explorer;

	using ObjectType = Engine::ObjectType;

	template<>
	class Explorer<ObjectType::Camera> : private ExplorerEditable
	{
	private:
		std::reference_wrapper<MultiProperties> mr_properties;
		std::reference_wrapper<Viewports> mr_viewports;
		RZ::Handle<RZ::Camera> m_selected, m_edited;
		Filter m_filter;

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
		Filter m_filter;

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
		RZ::SR::Handle<RZ::DirectLight> m_selected, m_edited;
		Filter m_filter;

	public:
		Explorer(std::reference_wrapper<MultiProperties> properties);

		void select(RZ::SR::Handle<RZ::DirectLight> selected);
		void update(RZ::World& world);
	};

	template <ObjectType T>
	concept MapObjectType =
		Utils::is::value<T>::template any_of<
		ObjectType::Texture,
		ObjectType::NormalMap,
		ObjectType::MetalnessMap,
		ObjectType::RoughnessMap,
		ObjectType::EmissionMap>::value;

	template<ObjectType T> requires MapObjectType<T>
	class Explorer<T> : private ExplorerEditable
	{
	public:
		using map_t = RZ::World::object_t<T>;
	private:
		std::reference_wrapper<MultiProperties> mr_properties;
		RZ::Handle<map_t> m_selected, m_edited;
		Filter m_filter;

	public:
		Explorer(std::reference_wrapper<MultiProperties> properties)
			: mr_properties(std::move(properties))
		{}
		void select(RZ::Handle<map_t> selected) { m_selected = selected; };
		void update(RZ::World& world)
		{
			m_filter.update();

			ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(0.0f, 3.0f));
			if (ImGui::BeginTable("map_table", 1, ImGuiTableFlags_BordersInnerH))
			{
				auto& maps = world.container<T>();
				for (uint32_t idx = 0; idx < maps.count(); idx++)
				{
					const auto& map = maps[idx];
					if (!map) continue;
					if (!m_filter.matches(map->name())) continue;

					ImGui::TableNextRow();
					ImGui::TableNextColumn();

					auto action = drawEditable(
						(map->name() + "##selectable_map" + std::to_string(idx)).c_str(),
						map == m_selected,
						map == m_edited);

					if (action.selected)
						m_selected = map;
					if (action.name_edited)
					{
						map->name(getEditedName());
						m_edited.release();
					}
					if (action.double_clicked)
					{
						m_edited = map;
						setNameToEdit(map->name());
					}

					const std::string popup_str_id = "map_popup" + std::to_string(idx);
					if (action.right_clicked)
						ImGui::OpenPopup(popup_str_id.c_str());
					if (ImGui::BeginPopup(popup_str_id.c_str()))
					{
						if (ImGui::Selectable("delete"))
							maps.destroy(map);
						ImGui::EndPopup();
					}
				}
				ImGui::EndTable();

				if (m_selected)
					mr_properties.get().setObject<T>(m_selected);
			}
			ImGui::PopStyleVar();
		}
	};

	template<>
	class Explorer<ObjectType::Material> : private ExplorerEditable
	{
	private:
		std::reference_wrapper<MultiProperties> mr_properties;
		RZ::Handle<RZ::Material> m_selected, m_edited;
		RZ::Material* mp_world_material = nullptr;
		Filter m_filter;

	public:
		Explorer(std::reference_wrapper<MultiProperties> properties);

		void select(RZ::Handle<RZ::Material> selected);
		void update(RZ::World& world);
	};

	template<>
	class Explorer<ObjectType::Mesh> : private ExplorerEditable
	{
	private:
		std::reference_wrapper<MultiProperties> mr_properties;
		RZ::Handle<RZ::Mesh> m_selected, m_edited;
		Filter m_filter;

	public:
		Explorer(std::reference_wrapper<MultiProperties> properties);

		void select(RZ::Handle<RZ::Mesh> selected);
		void update(RZ::World& world);
	};

	template<>
	class Explorer<ObjectType::Instance> : private ExplorerEditable
	{
	private:
		std::reference_wrapper<MultiProperties> mr_properties;
		RZ::Handle<RZ::Instance> m_selected_object, m_edited_object;
		RZ::Handle<RZ::Group> m_selected_group, m_edited_group;
		RZ::Handle<RZ::Group> m_group_to_delete;
		RZ::Handle<RZ::Instance> m_object_to_delete;
		Filter m_filter;
		bool m_delete_recursive = false;

		using drag_item_t = std::optional<std::variant<RZ::Handle<RZ::Instance>, RZ::Handle<RZ::Group>>>;
		using drop_item_t = std::optional<std::variant<std::monostate, RZ::Handle<RZ::Group>>>;
		drag_item_t m_drag_item;
		drop_item_t m_drop_item;
		static constexpr const char* sm_drag_drop_payload_id = "mesh_group_drag_drop_id";

	public:
		Explorer(std::reference_wrapper<MultiProperties> properties);

		void select(RZ::Handle<RZ::Instance> selected);
		void select(RZ::Handle<RZ::Group> selected);
		void update(RZ::World& world);
	private:
		void renderTree(const RZ::Handle<RZ::Group>& group, RZ::World& world);
		void renderObject(const RZ::Handle<RZ::Instance>& object, RZ::World& world);
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

			Explorer<ObjectType::Texture>,
			Explorer<ObjectType::NormalMap>,
			Explorer<ObjectType::MetalnessMap>,
			Explorer<ObjectType::RoughnessMap>,
			Explorer<ObjectType::EmissionMap>,

			Explorer<ObjectType::Material>,

			Explorer<ObjectType::Mesh>,
			Explorer<ObjectType::Instance>> m_explorers;
		bool m_selected = false;
		ObjectType m_selected_type{};

		bool m_opened = true;

	public:
		SceneExplorer(Scene& scene, Viewports& viewports);

		void open();
		void update();
		template <ObjectType T>
		void selectObject(const RZ::Handle<Engine::World::object_t<T>>& object)
		{
			std::get<Explorer<T>>(m_explorers).select(object);
			m_selected = true;
			m_selected_type = T;
		}
		template<>
		void selectObject<ObjectType::Group>(const RZ::Handle<RZ::Group>& group)
		{
			std::get<Explorer<ObjectType::Instance>>(m_explorers).select(group);
			m_selected = true;
			m_selected_type = ObjectType::Group;
		}
	};
}