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
		ImGui::EndTabBar();

		ImGui::End();

		m_properties.displayCurrentObject();
	}

	void Explorer::listCameras()
	{
		if (ImGui::CollapsingHeader("Cameras"))
		{
			static int current_item_idx = 0;

			ImGui::BeginListBox("list");
			auto& cameras = mr_scene.mr_world.Container<RZ::World::ContainerType::Camera>();
			for (size_t i = 0; i < cameras.GetCount(); i++)
			{
				auto& camera = cameras[i];
				const bool is_selected = (current_item_idx == i);
				if (ImGui::Selectable(camera->GetName().c_str(), is_selected))
				{
					current_item_idx = i;
					m_properties.setObject(camera);
				}

				if (is_selected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndListBox();
		}
	}
	void Explorer::listLights()
	{
		if (ImGui::CollapsingHeader("Spot Lights"))
		{
			static int current_item_idx = 0;

			ImGui::BeginListBox("list");
			auto& lights = mr_scene.mr_world.Container<RZ::World::ContainerType::SpotLight>();
			for (size_t light_idx = 0; light_idx < lights.GetCount(); light_idx++)
			{
				auto& light = lights[light_idx];
				const bool is_selected = (current_item_idx == light_idx);
				if (ImGui::Selectable(light->GetName().c_str(), is_selected))
				{
					current_item_idx = light_idx;
					m_properties.setObject(light);
				}

				if (is_selected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndListBox();
		}
		if (ImGui::CollapsingHeader("Direct lights"))
		{
			static int current_item_idx2 = 0;

			ImGui::BeginListBox("list2");
			auto& lights = mr_scene.mr_world.Container<RZ::World::ContainerType::DirectLight>();
			for (size_t light_idx = 0; light_idx < lights.GetCount(); light_idx++)
			{
				auto& light = lights[light_idx];
				const bool is_selected = (current_item_idx2 == light_idx);
				if (ImGui::Selectable(light->GetName().c_str(), is_selected))
				{
					current_item_idx2 = light_idx;
					m_properties.setObject(light);
				}

				if (is_selected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndListBox();
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
			ImGuiTreeNodeFlags_NoTreePushOnOpen |
			ImGuiTreeNodeFlags_SpanFullWidth);
		if (ImGui::IsItemClicked(ImGuiMouseButton_Left))
			m_properties.setObject(object);
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
			ImGuiTreeNodeFlags_SpanFullWidth);
		if (ImGui::IsItemClicked(ImGuiMouseButton_Left))
			m_properties.setObject(group);

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
			ImGuiTableFlags_BordersV))
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
		}
	}
}