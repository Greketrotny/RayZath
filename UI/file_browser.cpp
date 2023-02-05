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
			if (ImGui::Button("<", ImVec2(height, height)))
			{
				
			}
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("Previous location");

			ImGui::SameLine();
			if (ImGui::Button(">", ImVec2(height, height)))
			{
				
			}
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("Next location");

			ImGui::SameLine();
			if (ImGui::Button("^", ImVec2(height, height)))
			{
				if (auto parent_path = m_curr_path.parent_path(); parent_path != m_curr_path)
				{
					m_curr_path = std::move(parent_path);
					loadDirectoryContent();
				}
			}
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("parent folder");

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
		if (ImGui::BeginChild("TableChild", ImVec2(-1.0f, -height)))
		{
			ImGui::PushStyleColor(ImGuiCol_::ImGuiCol_WindowBg, ImVec4(0.0f, 1.0f, 0.0f, 0.0f));
			ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_ChildBorderSize, 10.0f);
			Complete complete_child([] {
			ImGui::PopStyleVar();
			ImGui::PopStyleColor(); 
				ImGui::EndChild();
				});
			if (ImGui::BeginTable("directory_content", 1, 0, ImVec2(-1.0f, -1.0f)))
			{
				bool directory_changed = false;
				Complete complete_table([] { ImGui::EndTable(); });
				for (const auto& item : m_directory_content)
				{
					ImGui::TableNextRow();
					ImGui::TableNextColumn();

					bool selected = false;
					if (item.is_directory())
					{
						if (ImGui::Selectable(
							("[D] " + item.path().filename().string()).c_str(), &selected,
							ImGuiSelectableFlags_::ImGuiSelectableFlags_DontClosePopups))
						{
							m_curr_path = item.path();
							directory_changed = true;
						}
					}
					else
					{
						if (ImGui::Selectable(
							("[F] " + item.path().filename().string()).c_str(), &selected,
							ImGuiSelectableFlags_::ImGuiSelectableFlags_DontClosePopups))
						{
							m_selected_file = item.path();
						}
					}					
				}
				if (directory_changed)
				{
					loadDirectoryContent();
				}
			}
		}
	}	

	void FileBrowserModal::loadDirectoryContent()
	{
		m_directory_content = std::vector<std::filesystem::directory_entry>(
			std::filesystem::directory_iterator(m_curr_path),
			std::filesystem::directory_iterator{});

		std::sort(
			m_directory_content.begin(), m_directory_content.end(),
			[](const auto& left, const auto& right)
			{
				return left.path() < right.path();
			});
		std::stable_sort(m_directory_content.begin(), m_directory_content.end(),
			[](const auto& left, const auto& right)
			{
				return left.is_directory() && !right.is_directory();
			});
	}
}
