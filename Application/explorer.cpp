#include "explorer.hpp"

#include "imgui.h"
#include "rayzath.hpp"

#include <ranges>

namespace RZ = RayZath::Engine;

namespace RayZath::UI::Windows
{
	using ObjectType = Engine::ObjectType;

	Explorer<ObjectType::Camera>::Explorer(
		std::reference_wrapper<MultiProperties> properties,
		std::reference_wrapper<Viewports> viewports)
		: mr_properties(std::move(properties))
		, mr_viewports(std::move(viewports))
	{}
	void Explorer<ObjectType::Camera>::select(RZ::Handle<RZ::Camera> selected)
	{
		m_selected = selected;
	}
	void Explorer<ObjectType::Camera>::update(RZ::World& world)
	{
		m_filter.update();

		ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(0.0f, 3.0f));
		if (ImGui::BeginTable("camera_table", 1, ImGuiTableFlags_BordersInnerH))
		{
			auto& cameras = world.container<RZ::ObjectType::Camera>();
			for (uint32_t idx = 0; idx < cameras.count(); idx++)
			{
				const auto& camera = cameras[idx];
				if (!camera) continue;
				if (!m_filter.matches(camera->name())) continue;

				ImGui::TableNextRow();
				ImGui::TableNextColumn();

				auto action = drawEditable(
					(camera->name() + "##selectable_camera" + std::to_string(idx)).c_str(),
					camera == m_selected,
					camera == m_edited);

				if (action.selected)
					m_selected = camera;
				if (action.name_edited)
				{
					camera->name(getEditedName());
					m_edited.release();
				}
				if (action.double_clicked)
				{
					m_edited = camera;
					setNameToEdit(camera->name());
				}

				const std::string popup_str_id = "spot_light_popup" + std::to_string(idx);
				if (action.right_clicked)
				{
					ImGui::OpenPopup(popup_str_id.c_str());
				}
				if (ImGui::BeginPopup(popup_str_id.c_str()))
				{
					if (ImGui::Selectable("show viewport"))
						mr_viewports.get().addViewport(camera);
					if (ImGui::Selectable("delete"))
						cameras.destroy(camera);
					if (ImGui::Selectable("duplicate"))
						cameras.create(RZ::ConStruct<RZ::Camera>(camera));
					ImGui::EndPopup();
				}
			}
			ImGui::EndTable();

			if (m_selected)
				mr_properties.get().setObject<ObjectType::Camera>(m_selected);
		}
		ImGui::PopStyleVar();
	}

	Explorer<ObjectType::SpotLight>::Explorer(std::reference_wrapper<MultiProperties> properties)
		: mr_properties(std::move(properties))
	{}
	void Explorer<ObjectType::SpotLight>::select(RZ::Handle<RZ::SpotLight> light)
	{
		m_selected = light;
	}
	void Explorer<ObjectType::SpotLight>::update(RZ::World& world)
	{
		m_filter.update();

		ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(0.0f, 3.0f));
		if (ImGui::BeginTable("spot_light_table", 1, ImGuiTableFlags_BordersInnerH))
		{
			auto& spot_lights = world.container<RZ::ObjectType::SpotLight>();
			for (uint32_t idx = 0; idx < spot_lights.count(); idx++)
			{
				const auto& light = spot_lights[idx];
				if (!light) continue;
				if (!m_filter.matches(light->name())) continue;

				ImGui::TableNextRow();
				ImGui::TableNextColumn();

				auto action = drawEditable(
					(light->name() + "##selectable_light" + std::to_string(idx)).c_str(),
					light == m_selected,
					light == m_edited);

				if (action.selected)
					m_selected = light;
				if (action.name_edited)
				{
					light->name(getEditedName());
					m_edited.release();
				}
				if (action.double_clicked)
				{
					m_edited = light;
					setNameToEdit(light->name());
				}

				const std::string popup_str_id = "spot_light_popup" + std::to_string(idx);
				if (action.right_clicked)
				{
					ImGui::OpenPopup(popup_str_id.c_str());
				}
				if (ImGui::BeginPopup(popup_str_id.c_str()))
				{
					if (ImGui::Selectable("delete"))
						spot_lights.destroy(light);
					if (ImGui::Selectable("duplicate"))
						spot_lights.create(RZ::ConStruct<RZ::SpotLight>(light));
					ImGui::EndPopup();
				}
			}
			ImGui::EndTable();

			if (m_selected)
				mr_properties.get().setObject<ObjectType::SpotLight>(m_selected);
		}
		ImGui::PopStyleVar();
	}

	Explorer<ObjectType::DirectLight>::Explorer(std::reference_wrapper<MultiProperties> properties)
		: mr_properties(std::move(properties))
	{}
	void Explorer<ObjectType::DirectLight>::select(RZ::SR::Handle<RZ::DirectLight> light)
	{
		m_selected = light;
	}
	void Explorer<ObjectType::DirectLight>::update(RZ::World& world)
	{
		m_filter.update();

		ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(0.0f, 3.0f));
		if (ImGui::BeginTable("direct_light_table", 1, ImGuiTableFlags_BordersInnerH))
		{
			auto lights{world.container<RZ::ObjectType::DirectLight>()};
			for (uint32_t idx = 0; idx < lights->count(); idx++)
			{
				auto& light = (*lights)[idx];
				if (!m_filter.matches(light.name())) continue;

				ImGui::TableNextRow();
				ImGui::TableNextColumn();

				auto action = drawEditable(
					(light.name() + "##selectable_light" + std::to_string(idx)).c_str(),
					m_selected == light,
					m_edited == light);

				if (action.selected)
					m_selected = lights->handle(idx);
				if (action.name_edited)
				{
					light.name(getEditedName());
					m_edited.reset();
				}
				if (action.double_clicked)
				{
					m_edited = lights->handle(light);
					setNameToEdit(light.name());
				}

				const std::string popup_str_id = "light_popup" + std::to_string(idx);
				if (action.right_clicked)
				{
					ImGui::OpenPopup(popup_str_id.c_str());
				}
				if (ImGui::BeginPopup(popup_str_id.c_str()))
				{
					if (ImGui::Selectable("delete"))
						lights->destroy(idx);
					if (ImGui::Selectable("duplicate"))
						lights->add(RZ::DirectLight(light));
					ImGui::EndPopup();
				}
			}
			ImGui::EndTable();

			if (m_selected)
				mr_properties.get().setObject<ObjectType::DirectLight>(m_selected);
		}
		ImGui::PopStyleVar();
	}

	Explorer<ObjectType::Material>::Explorer(std::reference_wrapper<MultiProperties> properties)
		: mr_properties(std::move(properties))
	{}
	void Explorer<ObjectType::Material>::select(RZ::Handle<RZ::Material> to_select)
	{
		m_selected = to_select;
	}
	void Explorer<ObjectType::Material>::update(RZ::World& world)
	{
		m_filter.update();

		ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(0.0f, 3.0f));
		if (ImGui::BeginTable("material_table", 1, ImGuiTableFlags_BordersInnerH))
		{
			// world material
			ImGui::TableNextRow();
			ImGui::TableNextColumn();

			if (ImGui::Selectable(
				"world material",
				mp_world_material != nullptr))
			{
				mp_world_material = &world.material();
				m_selected.release();
				m_edited.release();
			}

			auto& materials = world.container<RZ::ObjectType::Material>();
			for (uint32_t idx = 0; idx < materials.count(); idx++)
			{
				const auto& material = materials[idx];
				if (!material) continue;
				if (!m_filter.matches(material->name())) continue;

				ImGui::TableNextRow();
				ImGui::TableNextColumn();

				auto action = drawEditable(
					(material->name() + "##selectable_material" + std::to_string(idx)).c_str(),
					material == m_selected,
					material == m_edited);

				if (action.selected)
				{
					m_selected = material;
					mp_world_material = nullptr;
				}
				if (action.name_edited)
				{
					material->name(getEditedName());
					m_edited.release();
				}
				if (action.double_clicked)
				{
					m_edited = material;
					setNameToEdit(material->name());
				}

				const std::string popup_str_id = "material_popup" + std::to_string(idx);
				if (action.right_clicked)
				{
					ImGui::OpenPopup(popup_str_id.c_str());
				}
				if (ImGui::BeginPopup(popup_str_id.c_str()))
				{
					if (ImGui::Selectable("delete"))
						materials.destroy(material);
					if (ImGui::Selectable("duplicate"))
						materials.create(RZ::ConStruct<RZ::Material>(material));
					ImGui::EndPopup();
				}
			}
			ImGui::EndTable();

			if (mp_world_material)
				mr_properties.get().setObject<ObjectType::Material>(mp_world_material);
			else if (m_selected)
				mr_properties.get().setObject<ObjectType::Material>(m_selected);
		}
		ImGui::PopStyleVar();
	}

	Explorer<ObjectType::Mesh>::Explorer(std::reference_wrapper<MultiProperties> properties)
		: mr_properties(std::move(properties))
	{}
	void Explorer<ObjectType::Mesh>::select(RZ::Handle<RZ::Mesh> mesh)
	{
		m_selected = mesh;
	}
	void Explorer<ObjectType::Mesh>::update(RZ::World& world)
	{
		m_filter.update();

		ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(0.0f, 3.0f));
		if (ImGui::BeginTable("mesh_table", 1, ImGuiTableFlags_BordersInnerH))
		{
			auto& meshes = world.container<RZ::ObjectType::Mesh>();
			for (uint32_t idx = 0; idx < meshes.count(); idx++)
			{
				const auto& mesh = meshes[idx];
				if (!mesh) continue;
				if (!m_filter.matches(mesh->name())) continue;


				ImGui::TableNextRow();
				ImGui::TableNextColumn();

				auto action = drawEditable(
					(mesh->name() + "##selectable_mesh" + std::to_string(idx)).c_str(),
					mesh == m_selected,
					mesh == m_edited);

				if (action.selected)
					m_selected = mesh;
				if (action.name_edited)
				{
					mesh->name(getEditedName());
					m_edited.release();
				}
				if (action.double_clicked)
				{
					m_edited = mesh;
					setNameToEdit(mesh->name());
				}

				const std::string popup_str_id = "mesh_popup" + std::to_string(mesh.accessor()->idx());
				if (action.right_clicked)
					ImGui::OpenPopup(popup_str_id.c_str());
				if (ImGui::BeginPopup(popup_str_id.c_str()))
				{
					if (ImGui::Selectable("delete"))
						meshes.destroy(mesh);

					ImGui::EndPopup();
				}
			}
			ImGui::EndTable();

			if (m_selected)
				mr_properties.get().setObject<ObjectType::Mesh>(m_selected);
		}
		ImGui::PopStyleVar();
	}

	Explorer<ObjectType::Instance>::Explorer(std::reference_wrapper<MultiProperties> properties)
		: mr_properties(std::move(properties))
	{}
	void Explorer<ObjectType::Instance>::select(RZ::Handle<RZ::Instance> to_select)
	{
		m_selected_object = to_select;
		m_selected_group.release();
	}
	void Explorer<ObjectType::Instance>::select(RZ::Handle<RZ::Group> to_select)
	{
		m_selected_group = to_select;
		m_selected_object.release();
	}
	void Explorer<ObjectType::Instance>::update(RZ::World& world)
	{
		m_filter.update();

		// perform drag-drop operation
		if (m_drop_item && !m_drag_item) m_drop_item.reset();
		if (m_drag_item && m_drop_item)
		{
			if (const auto* const object = std::get_if<RZ::Handle<RZ::Instance>>(&*m_drag_item))
			{
				if (const auto* const group = std::get_if<RZ::Handle<RZ::Group>>(&*m_drop_item))
				{	// mesh on group
					RZ::Group::link(*group, *object);
				}
				if (std::get_if<std::monostate>(&*m_drop_item))
				{	// mesh on empty
					RZ::Group::unlink((*object)->group(), *object);
				}
			}
			if (const auto* const sub_group = std::get_if<RZ::Handle<RZ::Group>>(&*m_drag_item))
			{
				if (const auto* const group = std::get_if<RZ::Handle<RZ::Group>>(&*m_drop_item))
				{	// group on group
					RZ::Group::link(*group, *sub_group);
				}
				if (std::get_if<std::monostate>(&*m_drop_item))
				{	// group on empty
					RZ::Group::unlink((*sub_group)->group(), *sub_group);
				}
			}

			m_drop_item.reset();
			m_drag_item.reset();
		}

		if (ImGui::BeginChild("content table child", ImVec2(0, -1)))
		{
			if (ImGui::BeginTable("objects_table", 1, ImGuiTableFlags_BordersInnerH))
			{
				auto& groups = world.container<RZ::ObjectType::Group>();
				std::vector<RZ::Handle<RZ::Group>> root_groups;
				for (uint32_t idx = 0; idx < groups.count(); idx++)
				{
					auto& group = groups[idx];
					if (!group) continue;
					if (!group->group()) root_groups.push_back(group);
				}
				for (const auto& group : root_groups)
					renderTree(group, world);

				auto& objects = world.container<RZ::ObjectType::Instance>();
				for (uint32_t idx = 0; idx < objects.count(); idx++)
				{
					auto object = objects[idx];
					if (!object) continue;
					if (object->group()) continue;
					renderObject(object, world);
				}
				ImGui::EndTable();

				if (m_group_to_delete)
				{
					if (m_delete_recursive)
					{
						auto deleteRecursive = [&](
							const auto& deleteRecursiveFunc,
							const RZ::Handle<RZ::Group>& group) -> void
						{
							for (auto& object : group->objects())
								objects.destroy(object);
							for (auto& subgroup : group->groups())
								deleteRecursiveFunc(deleteRecursiveFunc, subgroup);
							groups.destroy(group);
						};
						deleteRecursive(deleteRecursive, m_group_to_delete);
					}
					else
					{
						if (m_group_to_delete->group())
						{
							for (auto& object : m_group_to_delete->objects())
								RZ::Group::link(m_group_to_delete->group(), object);
						}
						groups.destroy(m_group_to_delete);
					}
				}
				if (m_object_to_delete)
				{
					RZ::Group::unlink(m_object_to_delete->group(), m_object_to_delete);
					objects.destroy(m_object_to_delete);
				}

				if (m_selected_group)
					mr_properties.get().setObject<ObjectType::Group>(m_selected_group);
			}
		}
		ImGui::EndChild();

		// drop target for no group
		if (ImGui::BeginDragDropTarget())
		{
			if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(sm_drag_drop_payload_id);
				payload && !m_drop_item)
			{
				m_drop_item.emplace(std::monostate{}); // no group as a target (detach)
			}
			ImGui::EndDragDropTarget();
		}
	}
	void Explorer<ObjectType::Instance>::renderTree(const RZ::Handle<RZ::Group>& group, RZ::World& world)
	{
		if (!group) return;
		if (!m_filter.matches(group->name())) return;

		ImGui::TableNextRow();
		ImGui::TableNextColumn();

		// tree node (group)
		const bool open = ImGui::TreeNodeEx(
			(group->name() + "##group" + std::to_string(group.accessor()->idx())).c_str(),
			ImGuiTreeNodeFlags_SpanFullWidth |
			((m_selected_group == group) ? ImGuiTreeNodeFlags_Selected : 0) |
			ImGuiTreeNodeFlags_OpenOnArrow);
		if (ImGui::IsItemClicked(ImGuiMouseButton_Left))
		{
			m_selected_group = group;
			m_selected_object.release();
		}

		// group as drag/drop target
		if (ImGui::BeginDragDropTarget())
		{
			if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(
				sm_drag_drop_payload_id,
				ImGuiDragDropFlags_::ImGuiDragDropFlags_SourceAutoExpirePayload);
				payload && !m_drop_item)
			{
				m_drop_item = group;
			}
			ImGui::EndDragDropTarget();
		}
		if (ImGui::BeginDragDropSource())
		{
			m_drag_item.emplace(group);
			ImGui::SetDragDropPayload(sm_drag_drop_payload_id, nullptr, 0);
			ImGui::EndDragDropSource();
		}

		// group popup
		static uint32_t edit_id = std::numeric_limits<uint32_t>::max();
		static bool begin = false;
		const std::string popup_str_id = "group_popup_str_id" + std::to_string(group.accessor()->idx());
		const std::string rename_popup_id = "rename_popup_id" + std::to_string(group.accessor()->idx());
		if (ImGui::IsItemClicked(ImGuiMouseButton_Right))
			ImGui::OpenPopup(popup_str_id.c_str());
		if (ImGui::BeginPopup(popup_str_id.c_str()))
		{
			if (ImGui::MenuItem("delete"))
			{
				m_group_to_delete = group;
				m_delete_recursive = false;
			}
			if (ImGui::MenuItem("delete recursively"))
			{
				m_group_to_delete = group;
				m_delete_recursive = true;
			}
			if (ImGui::MenuItem("rename"))
			{
				begin = true;
				edit_id = group.accessor()->idx();
				setNameToEdit(group->name());
			}

			ImGui::EndPopup();
		}

		if (begin && edit_id == group.accessor()->idx())
		{
			ImGui::OpenPopup(rename_popup_id.c_str());
			begin = false;
		}
		if (ImGui::BeginPopup(rename_popup_id.c_str()))
		{
			auto action = drawEditable("new name", true, true, 300.0f);
			if (action.name_edited)
			{
				edit_id = std::numeric_limits<uint32_t>::max();
				group->name(getEditedName());
				m_edited_group.release();
				ImGui::CloseCurrentPopup();
			}
			ImGui::EndPopup();
		}

		// sub-group update
		if (open)
		{
			for (const auto& sub_group : group->groups())
				renderTree(sub_group, world);

			for (uint32_t idx = 0; idx < group->objects().size(); idx++)
				renderObject(group->objects()[idx], world);
			ImGui::TreePop();
		}
	}
	void Explorer<ObjectType::Instance>::renderObject(const RZ::Handle<RZ::Instance>& object, RZ::World& world)
	{
		if (!object) return;
		if (!m_filter.matches(object->name())) return;

		ImGui::TableNextRow();
		ImGui::TableNextColumn();

		// editable item
		auto action = drawEditable(
			(object->name() + "##selectable_object" + std::to_string(object.accessor()->idx())).c_str(),
			object == m_selected_object,
			object == m_edited_object);

		// object as drag/drop source
		if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_::ImGuiDragDropFlags_SourceAutoExpirePayload))
		{
			m_drag_item.emplace(object);
			ImGui::SetDragDropPayload(sm_drag_drop_payload_id, nullptr, 0);
			ImGui::EndDragDropSource();
		}

		if (action.selected)
		{
			m_selected_object = object;
			m_selected_group.release();
		}
		if (action.name_edited)
		{
			object->name(getEditedName());
			m_edited_object.release();
		}
		if (action.double_clicked)
		{
			m_edited_object = object;
			m_edited_group.release();
			setNameToEdit(object->name());
		}

		// popup
		const std::string popup_str_id = "object_popup" + std::to_string(object.accessor()->idx());
		if (action.right_clicked)
			ImGui::OpenPopup(popup_str_id.c_str());
		if (ImGui::BeginPopup(popup_str_id.c_str()))
		{
			auto& objects = world.container<ObjectType::Instance>();
			if (ImGui::Selectable("delete"))
			{
				m_object_to_delete = object;
			}
			if (ImGui::Selectable("duplicate"))
			{
				auto copy = objects.create(RZ::ConStruct<RZ::Instance>(object));
				RZ::Group::link(object->group(), copy);
			}

			ImGui::EndPopup();
		}

		if (m_selected_object)
			mr_properties.get().setObject<ObjectType::Instance>(m_selected_object);
	}

	SceneExplorer::SceneExplorer(Scene& scene, Viewports& viewports)
		: mr_scene(scene)
		, m_properties(mr_scene.mr_world)
		, m_viewports(viewports)
		, m_explorers{
			{std::ref(m_properties), std::ref(m_viewports)}, // camera

			std::ref(m_properties), // spot light
			std::ref(m_properties), // direct light

			std::ref(m_properties), // texture
			std::ref(m_properties), // normal map
			std::ref(m_properties), // metalness map
			std::ref(m_properties), // roughness map
			std::ref(m_properties), // emission map

			std::ref(m_properties), // material

			std::ref(m_properties), // mesh

			std::ref(m_properties) // instance
		}
	{}

	void SceneExplorer::open()
	{
		m_opened = true;
	}
	void SceneExplorer::update()
	{
		if (!m_opened) return;
		if (!ImGui::Begin("explorer", &m_opened))
		{
			ImGui::End();
			return;
		}

		ImGui::BeginTabBar("tabbar_world_objects",
			ImGuiTabBarFlags_Reorderable |
			ImGuiTabBarFlags_FittingPolicyScroll
		);

		if (const bool camera_selected = m_selected && m_selected_type == ObjectType::Camera;
			ImGui::BeginTabItem(
				"Cameras", nullptr,
				camera_selected ? ImGuiTabItemFlags_SetSelected : 0))
		{
			std::get<Explorer<ObjectType::Camera>>(m_explorers).update(mr_scene.mr_world);
			if (camera_selected) m_selected = false;
			ImGui::EndTabItem();
		}
		if (const bool object_selected =
			m_selected &&
			(m_selected_type == ObjectType::Instance || m_selected_type == ObjectType::Mesh);
			ImGui::BeginTabItem(
				"Objects", nullptr,
				object_selected ? ImGuiTabItemFlags_SetSelected : 0))
		{
			if (ImGui::BeginTabBar("tabbar_objects",
				ImGuiTabBarFlags_FittingPolicyResizeDown))
			{
				if (const bool instance_selected = m_selected && m_selected_type == ObjectType::Instance;
					ImGui::BeginTabItem(
						"instances", nullptr,
						instance_selected ? ImGuiTabItemFlags_SetSelected : 0))
				{
					std::get<Explorer<ObjectType::Instance>>(m_explorers).update(mr_scene.mr_world);
					if (instance_selected) m_selected = false;
					ImGui::EndTabItem();
				}
				if (const bool mesh_selected = m_selected && m_selected_type == ObjectType::Mesh;
					ImGui::BeginTabItem(
						"meshes", nullptr,
						mesh_selected ? ImGuiTabItemFlags_SetSelected : 0))
				{
					std::get<Explorer<ObjectType::Mesh>>(m_explorers).update(mr_scene.mr_world);
					if (mesh_selected) m_selected = false;
					ImGui::EndTabItem();
				}
				ImGui::EndTabBar();
			}
			ImGui::EndTabItem();
		}

		if (const bool lights_selected =
			m_selected &&
			(m_selected_type == ObjectType::SpotLight || m_selected_type == ObjectType::DirectLight);
			ImGui::BeginTabItem(
				"Lights", nullptr,
				lights_selected ? ImGuiTabItemFlags_SetSelected : 0))
		{
			if (ImGui::BeginTabBar("tabbar_lights",
				ImGuiTabBarFlags_FittingPolicyResizeDown))
			{
				if (const auto spot_selected = m_selected && m_selected_type == ObjectType::SpotLight;
					ImGui::BeginTabItem(
						"spot lights", nullptr,
						spot_selected ? ImGuiTabItemFlags_SetSelected : 0))
				{
					std::get<Explorer<ObjectType::SpotLight>>(m_explorers).update(mr_scene.mr_world);
					if (spot_selected) m_selected = false;
					ImGui::EndTabItem();
				}
				if (const auto direct_selected = m_selected && m_selected_type == ObjectType::DirectLight;
					ImGui::BeginTabItem(
						"direct lights", nullptr,
						direct_selected ? ImGuiTabItemFlags_SetSelected : 0))
				{
					std::get<Explorer<ObjectType::DirectLight>>(m_explorers).update(mr_scene.mr_world);
					if (direct_selected) m_selected = false;
					ImGui::EndTabItem();
				}
				ImGui::EndTabBar();
			}
			ImGui::EndTabItem();
		}
		if (const auto material_selected = m_selected && m_selected_type == ObjectType::Material;
			ImGui::BeginTabItem(
				"Materials", nullptr,
				m_selected && m_selected_type == ObjectType::Material ? ImGuiTabItemFlags_SetSelected : 0))
		{
			std::get<Explorer<ObjectType::Material>>(m_explorers).update(mr_scene.mr_world);
			if (material_selected) m_selected = false;
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Maps"))
		{
			using namespace std::string_view_literals;
			static constexpr std::array maps = {
				std::make_pair(ObjectType::Texture, "Texture"sv),
				std::make_pair(ObjectType::NormalMap, "Normal"sv),
				std::make_pair(ObjectType::MetalnessMap, "Metalness"sv),
				std::make_pair(ObjectType::RoughnessMap, "Roughness"sv),
				std::make_pair(ObjectType::EmissionMap, "Emission"sv)
			};

			if (ImGui::BeginTabBar("tabbar_maps",
				ImGuiTabBarFlags_FittingPolicyResizeDown))
			{
				for (const auto& [type, name] : maps)
				{
					if (const bool selected = m_selected && m_selected_type == type;
						ImGui::BeginTabItem(
							name.data(), nullptr,
							selected ? ImGuiTabItemFlags_SetSelected : 0))
					{
						switch (type)
						{
							case ObjectType::Texture:
								std::get<Explorer<ObjectType::Texture>>(m_explorers).update(mr_scene.mr_world);
								break;
							case ObjectType::NormalMap:
								std::get<Explorer<ObjectType::NormalMap>>(m_explorers).update(mr_scene.mr_world);
								break;
							case ObjectType::MetalnessMap:
								std::get<Explorer<ObjectType::MetalnessMap>>(m_explorers).update(mr_scene.mr_world);
								break;
							case ObjectType::RoughnessMap:
								std::get<Explorer<ObjectType::RoughnessMap>>(m_explorers).update(mr_scene.mr_world);
								break;
							case ObjectType::EmissionMap:
								std::get<Explorer<ObjectType::EmissionMap>>(m_explorers).update(mr_scene.mr_world);
								break;
							default: break;
						}
						if (selected) m_selected = false;
						ImGui::EndTabItem();
					}
				}
				ImGui::EndTabBar();
			}
			ImGui::EndTabItem();
		}
		ImGui::EndTabBar();
		ImGui::End();

		m_properties.displayCurrentObject();
	}
}
