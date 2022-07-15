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
	void NewModal<CommonMesh::Sphere>::update(Scene& scene)
	{
		using mesh_params_t = CommonMeshParameters<CommonMesh::Sphere>;

		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		if (m_opened)
			ImGui::OpenPopup("new sphere##new_sphere_modal_window");
		if (ImGui::BeginPopupModal(
			"new sphere##new_sphere_modal_window", &m_opened,
			ImGuiWindowFlags_AlwaysAutoResize))
		{
			using namespace std::string_view_literals;
			static const std::map names = {
				std::make_pair(mesh_params_t::Type::UVSphere, "uv sphere"sv),
				std::make_pair(mesh_params_t::Type::Icosphere, "icosphere"sv) };
			if (ImGui::BeginCombo("tesselation type", names.at(m_parameters.type).data()))
			{
				for (const auto& [type, name] : names)
				{
					if (ImGui::Selectable(
						name.data(),
						m_parameters.type == type))
						m_parameters.type = type;
				}
				ImGui::EndCombo();
			}

			switch (m_parameters.type)
			{
			case mesh_params_t::Type::UVSphere:
			{
				// resolution
				int resolution = int(m_parameters.resolution);
				if (ImGui::DragInt("resolution", &resolution, 0.1f, 4, std::numeric_limits<int>::max()))
					m_parameters.resolution = uint32_t(resolution);
				break;
			}
			case mesh_params_t::Type::Icosphere:
			{
				// subdivisions
				int subdivisions = int(m_parameters.resolution);
				if (ImGui::DragInt("subdivisions", &subdivisions, 0.1f, 0, std::numeric_limits<int>::max()))
					m_parameters.resolution = uint32_t(subdivisions);
				break;
			}
			}

			ImGui::Separator();
			
			ImGui::Checkbox("normals", &m_parameters.normals);
			ImGui::Checkbox("texture coordinates", &m_parameters.texture_coordinates);

			ImGui::SetNextItemWidth(-FLT_MIN);
			if (ImGui::Button("create", ImVec2(120, 0)))
			{
				m_opened = false;
				ImGui::CloseCurrentPopup();
				scene.Create<CommonMesh::Sphere>(m_parameters);
			}

			ImGui::EndPopup();
		}
	}
	void NewModal<CommonMesh::Cone>::update(Scene& scene)
	{
		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		if (m_opened)
			ImGui::OpenPopup("new cone##new_cone_modal_window");
		if (ImGui::BeginPopupModal(
			"new cone##new_cone_modal_window", &m_opened,
			ImGuiWindowFlags_AlwaysAutoResize))
		{
			int sides = int(m_parameters.side_faces);
			if (ImGui::DragInt("sides", &sides, 0.1f, 3, std::numeric_limits<int>::max()))
				m_parameters.side_faces = uint32_t(sides);

			ImGui::Separator();

			ImGui::Checkbox("normals", &m_parameters.normals);
			ImGui::Checkbox("texture coordinates", &m_parameters.texture_coordinates);

			if (ImGui::Button("create", ImVec2(120, 0)))
			{
				m_opened = false;
				ImGui::CloseCurrentPopup();
				scene.Create<CommonMesh::Cone>(m_parameters);
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
			// smooth shading
			ImGui::Checkbox("normals", &m_parameters.normals);

			if (ImGui::Button("create", ImVec2(120, 0)))
			{
				m_opened = false;
				ImGui::CloseCurrentPopup();
				scene.Create<CommonMesh::Cylinder>(m_parameters);
			}

			ImGui::EndPopup();
		}
	}
	void NewModal<CommonMesh::Torus>::update(Scene& scene)
	{
		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		if (m_opened)
			ImGui::OpenPopup("new torus##new_torus_modal_window");
		if (ImGui::BeginPopupModal(
			"new torus##new_torus_modal_window", &m_opened,
			ImGuiWindowFlags_AlwaysAutoResize))
		{
			// major resolution
			int major_resolution = int(m_parameters.major_resolution);
			if (ImGui::DragInt("major resolution", &major_resolution, 0.1f, 3, std::numeric_limits<int>::max()))
				m_parameters.major_resolution = uint32_t(major_resolution);
			// minor resolution
			int minor_resolution = int(m_parameters.minor_resolution);
			if (ImGui::DragInt("minor resolution", &minor_resolution, 0.1f, 3, std::numeric_limits<int>::max()))
				m_parameters.minor_resolution = uint32_t(minor_resolution);

			ImGui::DragFloat(
				"major radious", 
				&m_parameters.major_radious, 0.01f, 
				std::numeric_limits<float>::epsilon(), std::numeric_limits<float>::max(),
				"%.2f");
			ImGui::DragFloat(
				"minor radious",
				&m_parameters.minor_radious, 0.01f,
				std::numeric_limits<float>::epsilon(), std::numeric_limits<float>::max(),
				"%.2f");

			ImGui::Separator();

			ImGui::Checkbox("normals", &m_parameters.normals);
			ImGui::Checkbox("texture coordinates", &m_parameters.texture_coordinates);

			if (ImGui::Button("create", ImVec2(120, 0)))
			{
				m_opened = false;
				ImGui::CloseCurrentPopup();
				scene.Create<CommonMesh::Torus>(m_parameters);
			}

			ImGui::EndPopup();
		}
	}
}
