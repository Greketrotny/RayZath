#include "explorer.hpp"

#include "imgui.h"
#include "rayzath.h"

#include <ranges>
#include <iostream>

namespace RZ = RayZath::Engine;

namespace RayZath::UI::Windows
{
	using ObjectType = Engine::World::ObjectType;

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
		ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(0.0f, 3.0f));
		if (ImGui::BeginTable("camera_table", 1, ImGuiTableFlags_BordersInnerH))
		{
			auto& cameras = world.Container<RZ::World::ObjectType::Camera>();
			for (uint32_t idx = 0; idx < cameras.GetCount(); idx++)
			{
				const auto& camera = cameras[idx];
				if (!camera) continue;

				ImGui::TableNextRow();
				ImGui::TableNextColumn();

				auto action = drawEditable(
					(camera->GetName() + "##selectable_camera" + std::to_string(idx)).c_str(),
					camera == m_selected,
					camera == m_edited);

				if (action.selected)
					m_selected = camera;
				if (action.name_edited)
				{
					camera->SetName(getEditedName());
					m_edited.Release();
				}
				if (action.double_clicked)
				{
					m_edited = camera;
					setNameToEdit(camera->GetName());
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
						cameras.Destroy(camera);
					if (ImGui::Selectable("duplicate"))
						cameras.Create(RZ::ConStruct<RZ::Camera>(camera));
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
		ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(0.0f, 3.0f));
		if (ImGui::BeginTable("spot_light_table", 1, ImGuiTableFlags_BordersInnerH))
		{
			auto& spot_lights = world.Container<RZ::World::ObjectType::SpotLight>();
			for (uint32_t idx = 0; idx < spot_lights.GetCount(); idx++)
			{
				const auto& light = spot_lights[idx];
				if (!light) continue;

				ImGui::TableNextRow();
				ImGui::TableNextColumn();

				auto action = drawEditable(
					(light->GetName() + "##selectable_light" + std::to_string(idx)).c_str(),
					light == m_selected,
					light == m_edited);

				if (action.selected)
					m_selected = light;
				if (action.name_edited)
				{
					light->SetName(getEditedName());
					m_edited.Release();
				}
				if (action.double_clicked)
				{
					m_edited = light;
					setNameToEdit(light->GetName());
				}

				const std::string popup_str_id = "spot_light_popup" + std::to_string(idx);
				if (action.right_clicked)
				{
					ImGui::OpenPopup(popup_str_id.c_str());
				}
				if (ImGui::BeginPopup(popup_str_id.c_str()))
				{
					if (ImGui::Selectable("delete"))
						spot_lights.Destroy(light);
					if (ImGui::Selectable("duplicate"))
						spot_lights.Create(RZ::ConStruct<RZ::SpotLight>(light));
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
	void Explorer<ObjectType::DirectLight>::select(RZ::Handle<RZ::DirectLight> light)
	{
		m_selected = light;
	}
	void Explorer<ObjectType::DirectLight>::update(RZ::World& world)
	{
		ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(0.0f, 3.0f));
		if (ImGui::BeginTable("direct_light_table", 1, ImGuiTableFlags_BordersInnerH))
		{
			auto& lights = world.Container<RZ::World::ObjectType::DirectLight>();
			for (uint32_t idx = 0; idx < lights.GetCount(); idx++)
			{
				const auto& light = lights[idx];
				if (!light) continue;

				ImGui::TableNextRow();
				ImGui::TableNextColumn();

				auto action = drawEditable(
					(light->GetName() + "##selectable_light" + std::to_string(idx)).c_str(),
					light == m_selected,
					light == m_edited);

				if (action.selected)
					m_selected = light;
				if (action.name_edited)
				{
					light->SetName(getEditedName());
					m_edited.Release();
				}
				if (action.double_clicked)
				{
					m_edited = light;
					setNameToEdit(light->GetName());
				}

				const std::string popup_str_id = "spot_light_popup" + std::to_string(idx);
				if (action.right_clicked)
				{
					ImGui::OpenPopup(popup_str_id.c_str());
				}
				if (ImGui::BeginPopup(popup_str_id.c_str()))
				{
					if (ImGui::Selectable("delete"))
						lights.Destroy(light);
					if (ImGui::Selectable("duplicate"))
						lights.Create(RZ::ConStruct<RZ::DirectLight>(light));
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
				mp_world_material = &world.GetMaterial();
				m_selected.Release();
				m_edited.Release();
			}

			auto& materials = world.Container<RZ::World::ObjectType::Material>();
			for (uint32_t idx = 0; idx < materials.GetCount(); idx++)
			{
				const auto& material = materials[idx];
				if (!material) continue;

				ImGui::TableNextRow();
				ImGui::TableNextColumn();

				auto action = drawEditable(
					(material->GetName() + "##selectable_material" + std::to_string(idx)).c_str(),
					material == m_selected,
					material == m_edited);

				if (action.selected)
				{
					m_selected = material;
					mp_world_material = nullptr;
				}
				if (action.name_edited)
				{
					material->SetName(getEditedName());
					m_edited.Release();
				}
				if (action.double_clicked)
				{
					m_edited = material;
					setNameToEdit(material->GetName());
				}

				const std::string popup_str_id = "material_popup" + std::to_string(idx);
				if (action.right_clicked)
				{
					ImGui::OpenPopup(popup_str_id.c_str());
				}
				if (ImGui::BeginPopup(popup_str_id.c_str()))
				{
					if (ImGui::Selectable("delete"))
						materials.Destroy(material);
					if (ImGui::Selectable("duplicate"))
						materials.Create(RZ::ConStruct<RZ::Material>(material));
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

	Explorer<ObjectType::MeshStructure>::Explorer(std::reference_wrapper<MultiProperties> properties)
		: mr_properties(std::move(properties))
	{}
	void Explorer<ObjectType::MeshStructure>::select(RZ::Handle<RZ::MeshStructure> mesh)
	{
		m_selected = mesh;
	}
	void Explorer<ObjectType::MeshStructure>::update(RZ::World& world)
	{
		ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(0.0f, 3.0f));
		if (ImGui::BeginTable("mesh_table", 1, ImGuiTableFlags_BordersInnerH))
		{
			auto& meshes = world.Container<RZ::World::ObjectType::MeshStructure>();
			for (uint32_t idx = 0; idx < meshes.GetCount(); idx++)
			{
				const auto& mesh = meshes[idx];
				if (!mesh) continue;

				ImGui::TableNextRow();
				ImGui::TableNextColumn();

				auto action = drawEditable(
					(mesh->GetName() + "##selectable_mesh" + std::to_string(idx)).c_str(),
					mesh == m_selected,
					mesh == m_edited);

				if (action.selected)
					m_selected = mesh;
				if (action.name_edited)
				{
					mesh->SetName(getEditedName());
					m_edited.Release();
				}
				if (action.double_clicked)
				{
					m_edited = mesh;
					setNameToEdit(mesh->GetName());
				}

				const std::string popup_str_id = "mesh_popup" + std::to_string(mesh.GetAccessor()->GetIdx());
				if (action.right_clicked)
					ImGui::OpenPopup(popup_str_id.c_str());
				if (ImGui::BeginPopup(popup_str_id.c_str()))
				{
					if (ImGui::Selectable("delete"))
						meshes.Destroy(mesh);

					ImGui::EndPopup();
				}
			}
			ImGui::EndTable();

			if (m_selected)
				mr_properties.get().setObject<ObjectType::MeshStructure>(m_selected);
		}
		ImGui::PopStyleVar();
	}

	Explorer<ObjectType::Mesh>::Explorer(std::reference_wrapper<MultiProperties> properties)
		: mr_properties(std::move(properties))
	{}
	void Explorer<ObjectType::Mesh>::select(RZ::Handle<RZ::Mesh> to_select)
	{
		m_selected_object = to_select;
	}
	void Explorer<ObjectType::Mesh>::update(RZ::World& world)
	{
		std::ranges::fill(m_object_ids | std::views::values, false);
		std::ranges::fill(m_group_ids | std::views::values, false);

		if (ImGui::BeginTable("objects_table", 1,
			ImGuiTableFlags_BordersInnerH))
		{
			const auto& groups = world.Container<RZ::World::ObjectType::Group>();
			for (uint32_t idx = 0; idx < groups.GetCount(); idx++)
				renderTree(groups[idx], world);

			const auto& objects = world.Container<RZ::World::ObjectType::Mesh>();
			for (uint32_t idx = 0; idx < objects.GetCount(); idx++)
			{
				auto object = objects[idx];
				if (m_object_ids[object.GetAccessor()->GetIdx()]) continue;
				renderObject(object, world);
			}
			ImGui::EndTable();

			if (m_selected_group)
				mr_properties.get().setObject<ObjectType::Group>(m_selected_group);
		}
	}
	void Explorer<ObjectType::Mesh>::renderTree(const RZ::Handle<RZ::Group>& group, RZ::World& world)
	{
		if (!group) return;

		if (auto& already_drawn = m_group_ids[group.GetAccessor()->GetIdx()]; already_drawn) return;
		else already_drawn = true;

		ImGui::TableNextRow();
		ImGui::TableNextColumn();
		const bool open = ImGui::TreeNodeEx(
			(group->GetName() + "##group").c_str(),
			ImGuiTreeNodeFlags_SpanFullWidth |
			((m_selected_group == group) ? ImGuiTreeNodeFlags_Selected : 0) |
			ImGuiTreeNodeFlags_OpenOnArrow);
		if (ImGui::IsItemClicked(ImGuiMouseButton_Left))
		{
			m_selected_group = group;
			m_selected_object.Release();
		}

		if (open)
		{
			for (const auto& sub_group : group->groups())
				renderTree(sub_group, world);

			for (uint32_t idx = 0; idx < group->objects().size(); idx++)
				renderObject(group->objects()[idx], world);
			ImGui::TreePop();
		}
		else
		{
			for (const auto& object : group->objects())
			{
				if (auto& already_drawn = m_object_ids[object.GetAccessor()->GetIdx()]; already_drawn) continue;
				else already_drawn = true;
			}
		}
	}
	void Explorer<ObjectType::Mesh>::renderObject(const RZ::Handle<RZ::Mesh>& object, RZ::World& world)
	{
		if (!object) return;
		if (auto& already_drawn = m_object_ids[object.GetAccessor()->GetIdx()]; already_drawn) return;
		else already_drawn = true;

		ImGui::TableNextRow();
		ImGui::TableNextColumn();

		auto action = drawEditable(
			(object->GetName() + "##selectable_light" + std::to_string(object.GetAccessor()->GetIdx())).c_str(),
			object == m_selected_object,
			object == m_edited_object);

		if (action.selected)
		{
			m_selected_object = object;
			m_selected_group.Release();
		}
		if (action.name_edited)
		{
			object->SetName(getEditedName());
			m_edited_object.Release();
		}
		if (action.double_clicked)
		{
			m_edited_object = object;
			m_edited_group.Release();
			setNameToEdit(object->GetName());
		}

		const std::string popup_str_id = "object_popup" + std::to_string(object.GetAccessor()->GetIdx());
		if (action.right_clicked)
		{
			ImGui::OpenPopup(popup_str_id.c_str());
		}
		if (ImGui::BeginPopup(popup_str_id.c_str()))
		{
			auto& objects = world.Container<ObjectType::Mesh>();
			if (ImGui::Selectable("delete"))
			{
				RZ::Group::unlink(object->group(), object);
				objects.Destroy(object);
			}
			if (ImGui::Selectable("duplicate"))
			{
				auto copy = objects.Create(RZ::ConStruct<RZ::Mesh>(object));
				RZ::Group::link(object->group(), copy);
			}

			ImGui::EndPopup();
		}

		if (m_selected_object)
			mr_properties.get().setObject<ObjectType::Mesh>(m_selected_object);
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

			std::ref(m_properties), // mesh structure

			std::ref(m_properties) // mesh
	}
	{}
	void SceneExplorer::update()
	{
		ImGui::Begin("explorer", nullptr,
			ImGuiWindowFlags_NoCollapse);

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
			(m_selected_type == ObjectType::Mesh || m_selected_type == ObjectType::MeshStructure);
			ImGui::BeginTabItem(
				"Objects", nullptr,
				object_selected ? ImGuiTabItemFlags_SetSelected : 0))
		{
			if (ImGui::BeginTabBar("tabbar_objects",
				ImGuiTabBarFlags_FittingPolicyResizeDown))
			{
				if (const bool instance_selected = m_selected && m_selected_type == ObjectType::Mesh;
					ImGui::BeginTabItem(
						"instances", nullptr,
						instance_selected ? ImGuiTabItemFlags_SetSelected : 0))
				{
					std::get<Explorer<ObjectType::Mesh>>(m_explorers).update(mr_scene.mr_world);
					if (instance_selected) m_selected = false;
					ImGui::EndTabItem();
				}
				if (const bool mesh_selected = m_selected && m_selected_type == ObjectType::MeshStructure;
					ImGui::BeginTabItem(
						"meshes", nullptr,
						mesh_selected ? ImGuiTabItemFlags_SetSelected : 0))
				{
					std::get<Explorer<ObjectType::MeshStructure>>(m_explorers).update(mr_scene.mr_world);
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
