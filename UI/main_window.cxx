module;

#include "imgui.h"
#include "rayzath.h"

#include <iostream>

module rz.ui.windows.main;

import rz.ui.scene;

namespace RZ = RayZath::Engine;

namespace RayZath::UI::Windows
{
	Main::Main(Scene& scene)
		: mr_scene(scene)
	{}

	void Main::update()
	{
		ImGui::BeginMainMenuBar();

		if (ImGui::BeginMenu("new"))
		{
			if (ImGui::BeginMenu("light"))
			{
				if (ImGui::MenuItem("spot"))
				{
					auto& spot_lights = mr_scene.mr_world.Container<RZ::World::ContainerType::SpotLight>();
					spot_lights.Create(RZ::ConStruct<RZ::SpotLight>("new spot light"));
				}
				if (ImGui::MenuItem("direct"))
				{
					auto& direct_lights = mr_scene.mr_world.Container<RZ::World::ContainerType::DirectLight>();
					direct_lights.Create(RZ::ConStruct<RZ::DirectLight>("new direct light"));
				}
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("object"))
			{
				ImGui::EndMenu();
			}
			ImGui::EndMenu();
		}

		ImGui::EndMainMenuBar();
	}
}
