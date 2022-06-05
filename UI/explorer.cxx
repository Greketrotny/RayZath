module;

#include "imgui.h"

module rz.ui.windows.explorer;

namespace RayZath::UI::Windows
{
	void Explorer::update()
	{
		ImGui::Begin("explorer", nullptr,
			ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse |
			ImGuiWindowFlags_::ImGuiWindowFlags_HorizontalScrollbar);

		ImGui::End();
	}
}