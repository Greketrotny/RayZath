#include "save_modals.hpp"

#include "imgui.h"

namespace RayZath::UI::Windows
{
	void SceneSaveModal::update(Scene& scene)
	{
		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constinit auto* popup_id = "save scene##save_scene_modal_window";
		if (m_opened) ImGui::OpenPopup(popup_id);
		if (ImGui::BeginPopupModal(popup_id, &m_opened, ImGuiWindowFlags_AlwaysAutoResize))
		{
			const auto width = 300.0f;
			// path
			ImGui::SetNextItemWidth(width);
			const bool completed = ImGui::InputTextWithHint("##scene_name_input", "name",
				m_path_buffer.data(), m_path_buffer.size(), ImGuiInputTextFlags_EnterReturnsTrue);
			// allow partial write
			ImGui::Checkbox("allow partial write", &m_save_options.allow_partial_write);

			ImGui::SetNextItemWidth(-1.0f);
			if (ImGui::Button("save", ImVec2(50, 0)) || completed)
			{
				try
				{
					m_save_options.path = std::filesystem::path(m_path_buffer.data());
					scene.mr_world.GetSaver().SaveScene(m_save_options);
					ImGui::CloseCurrentPopup();
					m_opened = false;
				}
				catch (RayZath::Exception& e)
				{
					m_fail_message = e.ToString();
				}
				catch (std::exception& e)
				{
					m_fail_message = e.what();
				}
			}

			if (m_fail_message)
			{
				ImGui::SetNextItemWidth(width);
				ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 64, 64, 255));
				ImGui::TextWrapped(("Failed to save scene at specified path. Reason: " + *m_fail_message).c_str());
				ImGui::PopStyleColor();
			}
			ImGui::EndPopup();
		}
	}
}
