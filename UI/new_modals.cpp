#include "new_modals.hpp"

#include "imgui.h"

#include <iostream>

namespace RayZath::UI::Windows
{	
	void NewModal<CommonMesh::Plane>::update(Scene& scene)
	{
		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
		
		if (m_opened)
			ImGui::OpenPopup("new plane##new_plane_modal_window");
		if (ImGui::BeginPopupModal(
			"new plane##new_plane_modal_window", &m_opened,
			ImGuiWindowFlags_AlwaysAutoResize))
		{
			ImGui::DragFloat("width", &m_parameters.width, 0.01f, 0.0001f, std::numeric_limits<float>::max(), "%.2f");
			ImGui::DragFloat("height", &m_parameters.height, 0.01f, 0.0001f, std::numeric_limits<float>::max(), "%.2f");
			int sides = int(m_parameters.sides);
			if (ImGui::DragInt("sides", &sides, 0.1f, 3, std::numeric_limits<int>::max()))
				m_parameters.sides = uint32_t(sides);

			if (ImGui::Button("create", ImVec2(120, 0)))
			{
				m_opened = false;
				ImGui::CloseCurrentPopup();
				scene.Create<CommonMesh::Plane>(m_parameters);
			}
			ImGui::EndPopup();
		}
	}
	void NewModal<CommonMesh::Cylinder>::update(Scene& scene)
	{
		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		if (m_opened)
			ImGui::OpenPopup("new cylinder##new_cylinder_modal_window");
		if (ImGui::BeginPopupModal(
			"new cylinder##new_cylinder_modal_window", &m_opened,
			ImGuiWindowFlags_AlwaysAutoResize))
		{
			// face count
			int faces = int(m_parameters.faces);
			if (ImGui::DragInt("faces", &faces, 0.1f, 3, std::numeric_limits<int>::max()))
				m_parameters.faces = uint32_t(faces);
			// shooth shading
			ImGui::Checkbox("smooth shading", &m_parameters.smooth_shading);

			if (ImGui::Button("create", ImVec2(120, 0)))
			{
				m_opened = false;
				ImGui::CloseCurrentPopup();
				scene.Create<CommonMesh::Cylinder>(m_parameters);
			}

			ImGui::EndPopup();
		}
	}
}
