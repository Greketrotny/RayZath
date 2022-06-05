module;

#include "imgui.h"

#include "rayzath.h"

#include <iostream>

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
			ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse |
			ImGuiWindowFlags_::ImGuiWindowFlags_HorizontalScrollbar);

		ImGuiTabItemFlags_::ImGuiTabItemFlags_NoCloseWithMiddleMouseButton;
		ImGui::BeginTabBar("tabbar_objects", ImGuiTabBarFlags_Reorderable);
		if (ImGui::BeginTabItem("Cameras"))
		{
			ImGui::Text("list of cameras");
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Objects"))
		{
			ImGui::Text("list of objects");
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Lights"))
		{
			listLights();
			ImGui::EndTabItem();
		}
		ImGui::EndTabBar();

		ImGui::End();
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
					current_item_idx = light_idx;

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
					current_item_idx2 = light_idx;

				if (is_selected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndListBox();
		}
	}
}