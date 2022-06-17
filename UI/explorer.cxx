module;

#include "imgui.h"
#include "rayzath.h"

#include <iostream>
#include <ranges>

module rz.ui.windows.explorer;

import rz.ui.scene;

namespace RZ = RayZath::Engine;

namespace RayZath::UI::Windows
{
	Explorer::Explorer(Scene& scene)
		: mr_scene(scene)
	{}

	void Explorer::update()
	{
		ImGui::Begin("explorer", nullptr,
			ImGuiWindowFlags_NoCollapse |
			ImGuiWindowFlags_HorizontalScrollbar);

		ImGuiTabItemFlags_::ImGuiTabItemFlags_NoCloseWithMiddleMouseButton;
		ImGui::BeginTabBar("tabbar_objects", ImGuiTabBarFlags_Reorderable);
		if (ImGui::BeginTabItem("Cameras"))
		{
			listCameras();
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Objects"))
		{
			listObjects();
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Lights"))
		{
			listLights();
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Materials"))
		{
			listMaterials();
			ImGui::EndTabItem();
		}
		ImGui::EndTabBar();

		ImGui::End();

		m_properties.displayCurrentObject();
	}

	void Explorer::listCameras()
	{
		if (ImGui::BeginTable("spot_light_table", 1, ImGuiTableFlags_BordersInnerH))
		{
			static RZ::Handle<RZ::Camera> current_camera;
			const auto& cameras = mr_scene.mr_world.Container<RZ::World::ContainerType::Camera>();
			for (uint32_t idx = 0; idx < cameras.GetCount(); idx++)
			{
				const auto& camera = cameras[idx];
				if (!camera) continue;

				ImGui::TableNextRow();
				ImGui::TableNextColumn();

				if (ImGui::Selectable(
					camera->GetName().c_str(),
					camera == current_camera))
					current_camera = camera;
			}
			ImGui::EndTable();

			if (current_camera)
			{
				m_properties.setObject(current_camera);
			}
		}
	}
	void Explorer::listLights()
	{
		static RZ::Handle<RZ::SpotLight> current_spot_light;
		static RZ::Handle<RZ::DirectLight> current_direct_light;

		if (ImGui::CollapsingHeader("Spot Lights"))
		{
			ImGui::Indent();
			if (ImGui::BeginTable("spot_light_table", 1, ImGuiTableFlags_BordersInnerH))
			{
				const auto& spot_lights = mr_scene.mr_world.Container<RZ::World::ContainerType::SpotLight>();
				for (uint32_t idx = 0; idx < spot_lights.GetCount(); idx++)
				{
					const auto& light = spot_lights[idx];
					if (!light) continue;

					ImGui::TableNextRow();
					ImGui::TableNextColumn();

					if (ImGui::Selectable(
						light->GetName().c_str(),
						light == current_spot_light))
						current_spot_light = light;
				}
				ImGui::EndTable();

				if (current_spot_light)
				{
					current_direct_light.Release();
					m_properties.setObject(current_spot_light);
				}
			}
			ImGui::Unindent();
		}
		if (ImGui::CollapsingHeader("Direct lights"))
		{
			ImGui::Indent();
			if (ImGui::BeginTable("direct_light_table", 1, ImGuiTableFlags_BordersInnerH))
			{
				const auto& direct_lights = mr_scene.mr_world.Container<RZ::World::ContainerType::DirectLight>();
				for (uint32_t idx = 0; idx < direct_lights.GetCount(); idx++)
				{
					const auto& light = direct_lights[idx];
					if (!light) continue;

					ImGui::TableNextRow();
					ImGui::TableNextColumn();

					if (ImGui::Selectable(
						light->GetName().c_str(),
						light == current_direct_light))
						current_direct_light = light;
				}
				ImGui::EndTable();

				if (current_direct_light)
				{
					current_spot_light.Release();
					m_properties.setObject(current_direct_light);
				}
			}
			ImGui::Unindent();
		}
	}

	void Explorer::listObject(const RZ::Handle<RZ::Mesh>& object)
	{
		if (auto& already_drawn = m_object_ids[object.GetAccessor()->GetIdx()]; already_drawn) return;
		else already_drawn = true;

		ImGui::TableNextRow();
		ImGui::TableNextColumn();
		ImGui::TreeNodeEx((object->GetName() + "##object").c_str(),
			ImGuiTreeNodeFlags_Leaf |
			((m_current_object == object) ? ImGuiTreeNodeFlags_Selected : 0) |
			ImGuiTreeNodeFlags_NoTreePushOnOpen |
			ImGuiTreeNodeFlags_SpanFullWidth);
		if (ImGui::IsItemClicked(ImGuiMouseButton_Left))
		{
			m_current_group.Release();
			m_current_object = object;
		}

		if (m_current_object)
			m_properties.setObject(m_current_object);
	}
	void Explorer::objectTree(const RZ::Handle<RZ::Group>& group)
	{
		if (!group) return;

		if (auto& already_drawn = m_group_ids[group.GetAccessor()->GetIdx()]; already_drawn) return;
		else already_drawn = true;

		ImGui::TableNextRow();
		ImGui::TableNextColumn();
		const bool open = ImGui::TreeNodeEx(
			(group->GetName() + "##group").c_str(),
			ImGuiTreeNodeFlags_SpanFullWidth |
			((m_current_group == group) ? ImGuiTreeNodeFlags_Selected : 0) |
			ImGuiTreeNodeFlags_OpenOnArrow);
		if (ImGui::IsItemClicked(ImGuiMouseButton_Left))
		{
			m_current_group = group;
			m_current_object.Release();
		}

		if (open)
		{
			for (const auto& sub_group : group->groups())
				objectTree(sub_group);

			for (const auto& object : group->objects())
				listObject(object);
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
	void Explorer::listObjects()
	{
		std::ranges::fill(m_object_ids | std::views::values, false);
		std::ranges::fill(m_group_ids | std::views::values, false);

		if (ImGui::BeginTable("objects_table", 1,
			ImGuiTableFlags_BordersInnerH))
		{
			const auto& groups = mr_scene.mr_world.Container<RZ::World::ContainerType::Group>();
			for (uint32_t idx = 0; idx < groups.GetCount(); idx++)
				objectTree(groups[idx]);

			const auto& objects = mr_scene.mr_world.Container<RZ::World::ContainerType::Mesh>();
			for (uint32_t idx = 0; idx < objects.GetCount(); idx++)
			{
				const auto& object = objects[idx];
				if (m_object_ids[object.GetAccessor()->GetIdx()]) continue;
				listObject(object);
			}
			ImGui::EndTable();

			if (m_current_group)
				m_properties.setObject(m_current_group);
		}
	}

	void Explorer::listMaterials()
	{
		if (ImGui::BeginTable("materials_table", 1, ImGuiTableFlags_BordersInnerH))
		{
			static RZ::Handle<RZ::Material> current_material;

			const auto& materials = mr_scene.mr_world.Container<RZ::World::ContainerType::Material>();
			for (uint32_t idx = 0; idx < materials.GetCount(); idx++)
			{
				const auto& material = materials[idx];
				if (!material) continue;

				ImGui::TableNextRow();
				ImGui::TableNextColumn();

				if (ImGui::Selectable(
					material->GetName().c_str(),
					material == current_material))
					current_material = material;
			}
			ImGui::EndTable();

			if (current_material)
				m_properties.setObject(current_material);
		}
	}
}