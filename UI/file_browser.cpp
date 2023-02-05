#include "file_browser.hpp"

#include "imgui.h"

#include <functional>
#include <type_traits>

#include <iostream>
#include <string.h>

ImVec2 operator/(const ImVec2& v, float a)
{
	return ImVec2(v.x / a, v.y / a);
}

namespace RayZath::UI::Windows
{
	template <typename F>
	struct Complete
	{
	private:
		F m_f;
	public:
		Complete(F f) 
			: m_f(std::move(f)) 
		{}
		~Complete() noexcept(std::is_nothrow_invocable_v<F>)
		{
			m_f();
		}
	};

	FileBrowserModal::FileBrowserModal(std::filesystem::path start_path)
		: m_curr_path(std::move(start_path))
	{
		loadDirectoryContent();
	}

	bool FileBrowserModal::render()
	{
		bool selected = false;
		ImGui::SetNextWindowPos(
			ImGui::GetMainViewport()->GetCenter(), 
			ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
		//ImGui::SetNextWindowSize(ImGui::GetMainViewport()->Size / 2.0f);
		
		static constexpr auto* popup_id = "File Browser##load_texture_modal_window";
		if (m_opened) ImGui::OpenPopup(popup_id);
		if (ImGui::BeginPopupModal(popup_id, &m_opened))
		{
			Complete complete_popup([] { ImGui::EndPopup(); });

			const auto height = ImGui::GetFrameHeight();
			ImGui::Button("<", ImVec2(height, height));
			ImGui::SameLine();
			ImGui::Button(">", ImVec2(height, height));
			ImGui::SameLine();
			ImGui::Button("^", ImVec2(height, height));
			ImGui::SameLine();
			const auto path_str = m_curr_path.string();
			strncpy_s(m_path_buff.data(), m_path_buff.size(), path_str.c_str(), path_str.size());

			ImGui::SetNextItemWidth(-1.0f);
			ImGui::InputText("##input", m_path_buff.data(), m_path_buff.size());

			directoryContent();		

			if (ImGui::Button("Select"))
			{
				selected = true;
				ImGui::CloseCurrentPopup();
				m_opened = false;
			}
		}

		return selected;
	}
	void FileBrowserModal::directoryContent()
	{
		const auto height = ImGui::GetFrameHeightWithSpacing();
		ImGui::PushStyleColor(ImGuiCol_::ImGuiCol_WindowBg, ImVec4(0.0f, 0.1f, 0.0f, 0.0f));
		if (ImGui::BeginChild("TableChild", ImVec2(-1.0f, -height)))
		{
			Complete complete_child([] { ImGui::EndChild(); ImGui::PopStyleColor(); });
			if (ImGui::BeginTable("directory_content", 1, 0, ImVec2(-1.0f, -1.0f)))
			{
				Complete complete_table([] { ImGui::EndTable(); });
				for (const auto& item : m_directory_content)
				{
					ImGui::TableNextRow();
					ImGui::TableNextColumn();

					bool selected = false;
					if (ImGui::Selectable(item.path().filename().string().c_str(), &selected,
						ImGuiSelectableFlags_::ImGuiSelectableFlags_DontClosePopups))
					{
						m_selected_file = item.path();
					}
				}
			}
		}
	}	

	void FileBrowserModal::loadDirectoryContent()
	{
		m_directory_content = std::vector<std::filesystem::directory_entry>(
			std::filesystem::directory_iterator(m_curr_path),
			std::filesystem::directory_iterator{});
	}
}
