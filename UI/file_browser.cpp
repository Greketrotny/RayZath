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
			Complete complete_child([] { ImGui::PopStyleVar(); ImGui::PopStyleColor(); ImGui::EndChild(); });
			
			if (ImGui::BeginTable("directory_content", 1, 0, ImVec2(-1.0f, -1.0f)))
			{
				bool directory_changed = false;
				Complete complete_table([] { ImGui::EndTable(); });
				for (size_t item_idx = 0; item_idx < m_directory_content.size(); item_idx++)
				{
					const auto& item = m_directory_content[item_idx];

					ImGui::TableNextRow();
					ImGui::TableNextColumn();

					bool selected = m_selected_items.contains(item_idx);
					std::string prefix = item.is_directory() ? "[D] " : "[F]";
					if (ImGui::Selectable(
						(prefix + item.path().filename().string()).c_str(), &selected,
						ImGuiSelectableFlags_::ImGuiSelectableFlags_DontClosePopups))
					{
						const bool ctrl = ImGui::GetIO().KeyCtrl;
						const bool shift = ImGui::GetIO().KeyShift;

						if (shift)
						{
							m_selected_items.clear();
							for (size_t i = std::min(item_idx, m_last_clicked);
								i <= std::max(item_idx, m_last_clicked);
								i++)
							{
								m_selected_items.insert(i);
							}
						}
						else if (ctrl)
						{
							if (const auto it = m_selected_items.find(item_idx);
								it == m_selected_items.end())
							{
								m_selected_items.insert(item_idx);
							}
							else { m_selected_items.erase(item_idx); }
							m_last_clicked = item_idx;
						}
						else
						{
							if (item.is_directory())
							{
								directory_changed = true;
								m_curr_path = item.path();
							}
							else
							{
								m_selected_items.clear();
								if (const auto it = m_selected_items.find(item_idx);
									it == m_selected_items.end())
								{
									m_selected_items.insert(item_idx);
								}
								else { m_selected_items.erase(item_idx); }
								m_last_clicked = item_idx;
							}
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
		m_selected_items.clear();
		m_last_clicked = 0;
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

	std::vector<std::filesystem::path> FileBrowserModal::selectedFiles()
	{
		std::vector<std::filesystem::path> selected_files;
		for (const auto& file_idx : m_selected_items)
			if (const auto& entry = m_directory_content[file_idx]; !entry.is_directory())
				selected_files.push_back(entry.path());
		return selected_files;
	}
}
