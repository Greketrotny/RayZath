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
	FileBrowserModal::FileBrowserModal(std::filesystem::path start_path, Mode mode)
		: m_mode(mode)
		, m_path_history{std::move(start_path)}
		, m_curr_path(m_path_history.begin())
	{
		loadDirectoryContent();
	}

	bool FileBrowserModal::render()
	{
		bool confirmed = false;
		try
		{
			ImGui::SetNextWindowPos(
				ImGui::GetMainViewport()->GetCenter(),
				ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

			static constexpr auto* popup_id = "File Browser##file_browser_modal_window";
			if (m_opened) ImGui::OpenPopup(popup_id);
			if (ImGui::BeginPopupModal(popup_id, &m_opened))
			{
				Complete complete_popup([] { ImGui::EndPopup(); });

				renderNavButtons();
				renderPathBar();
				renderDirectoryContent();
				confirmed = renderBottomBar();
				if (confirmed)
				{
					ImGui::CloseCurrentPopup();
					m_opened = false;
				}

				m_message_box.render();
			}
		}
		catch (std::filesystem::filesystem_error& e)
		{
			m_message_box = MessageBox(e.what(), {"Ok"});
		}

		return confirmed;
	}
	void FileBrowserModal::renderNavButtons()
	{
		const auto height = ImGui::GetFrameHeight();
		if (ImGui::Button("<", ImVec2(height, height)))
		{
			if (m_curr_path != m_path_history.begin())
			{
				--m_curr_path;
				loadDirectoryContent();
			}
		}
		if (ImGui::IsItemHovered())
			ImGui::SetTooltip("Previous location");

		ImGui::SameLine();
		if (ImGui::Button(">", ImVec2(height, height)))
		{
			if (m_curr_path != std::prev(m_path_history.end()))
			{
				++m_curr_path;
				loadDirectoryContent();
			}
		}
		if (ImGui::IsItemHovered())
			ImGui::SetTooltip("Next location");

		ImGui::SameLine();
		if (ImGui::Button("^", ImVec2(height, height)))
		{
			if (auto parent_path = m_curr_path->parent_path(); parent_path != *m_curr_path)
			{
				setCurrentPath(std::move(parent_path));
				loadDirectoryContent();
			}
		}
		if (ImGui::IsItemHovered())
			ImGui::SetTooltip("Parent folder");

		if (m_mode == Mode::Save)
		{
			ImGui::SameLine();
			if (ImGui::Button("+", ImVec2(height, height)))
			{
				m_adding_new_folder = true;
			}
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("New folder");
		}		

		ImGui::SameLine();
	}
	void FileBrowserModal::renderPathBar()
	{
		ImGui::SetNextItemWidth(-1.0f);
		if (ImGui::InputText("##input", m_path_buff.data(), m_path_buff.size(),
			ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll))
		{
			std::filesystem::path entered_path(m_path_buff.data());
			if (std::filesystem::exists(entered_path))
			{
				setCurrentPath(std::move(entered_path));
				loadDirectoryContent();
			}
			else
			{
				m_message_box = MessageBox(
					"Entered path doesn't exist. Create one?",
					{"Yes", "No"},
					[this, entered_path](MessageBox::option_t option) mutable {
						if (!option) return;

				if (*option == "No") return;
				if (*option == "Yes")
				{
					std::filesystem::create_directories(entered_path);
					setCurrentPath(std::move(entered_path));
					loadDirectoryContent();
				}
					});
			}
		}
	}
	void FileBrowserModal::renderDirectoryContent()
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

				if (m_mode == Mode::Save && m_adding_new_folder)
				{
					ImGui::TableNextRow();
					ImGui::TableNextColumn();
					renderNewFolderInput();
				}

				for (size_t item_idx = 0; item_idx < m_directory_content.size(); item_idx++)
				{
					const auto& item = m_directory_content[item_idx];

					ImGui::TableNextRow();
					ImGui::TableNextColumn();

					bool selected = m_selected_items.contains(item_idx);
					std::string name = item.is_directory() ? "[D] " : "[F] ";
					name += item.path().filename().string();
					if (ImGui::Selectable(
						name.c_str(), &selected,
						ImGuiSelectableFlags_DontClosePopups))
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
								setCurrentPath(item.path());
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
	void FileBrowserModal::renderNewFolderInput()
	{
		ImGui::Text("[D] ");
		ImGui::SameLine();
		ImGui::SetNextItemWidth(-1.0f);
		if (ImGui::InputText("##new_folder_input", m_new_folder_buff.data(), m_new_folder_buff.size(),
			ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll))
		{
			std::filesystem::create_directory(*m_curr_path / std::filesystem::path(m_new_folder_buff.data()));
			loadDirectoryContent();
			m_adding_new_folder = false;
		}
	}
	bool FileBrowserModal::renderBottomBar()
	{
		if (m_mode == Mode::Open)
		{
			if (ImGui::Button("Select"))
				return true;
		}
		else
		{
			bool confirmed = false;
			if (ImGui::Button("Confirm"))
			{
				confirmed = true;
			}
			ImGui::SameLine();
			ImGui::SetNextItemWidth(-1.0f);
			if (ImGui::InputText("##file_input", m_file_buff.data(), m_file_buff.size(),
				ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll))
			{
				confirmed = true;
			}

			if (confirmed)
			{
				std::string entered_file(m_file_buff.data());
				if (entered_file.empty())
				{
					m_message_box = MessageBox(
						"Enter non-empty file name",
						{});
					return false;
				}
				return true;
			}
		}
		return false;
	}

	void FileBrowserModal::loadDirectoryContent()
	{
		auto new_content = std::vector<std::filesystem::directory_entry>(
			std::filesystem::directory_iterator(*m_curr_path),
			std::filesystem::directory_iterator{});
		std::sort(
			new_content.begin(), new_content.end(),
			[](const auto& left, const auto& right)
			{
				return left.path() < right.path();
			});
		std::stable_sort(new_content.begin(), new_content.end(),
			[](const auto& left, const auto& right)
			{
				return left.is_directory() && !right.is_directory();
			});

		strncpy_s(
			m_path_buff.data(), m_path_buff.size(),
			m_curr_path->string().c_str(), m_curr_path->string().size());

		m_directory_content = std::move(new_content);
		m_selected_items.clear();
		m_last_clicked = 0;
		m_adding_new_folder = false;
	}
	void FileBrowserModal::setCurrentPath(std::filesystem::path new_path)
	{
		m_path_history.erase(std::next(m_curr_path), m_path_history.end());
		m_path_history.push_back(std::move(new_path));
		if (m_path_history.size() > 32u)
		{
			m_path_history.erase(
				m_path_history.begin(),
				std::next(m_path_history.begin(), m_path_history.size() - 25u));
		}
		m_curr_path = std::prev(m_path_history.end());
	}

	std::vector<std::filesystem::path> FileBrowserModal::selectedFiles()
	{
		if (m_mode == Mode::Open)
		{
			std::vector<std::filesystem::path> selected_files;
			for (const auto& file_idx : m_selected_items)
				if (const auto& entry = m_directory_content[file_idx]; !entry.is_directory())
					selected_files.push_back(entry.path());
			return selected_files;
		}
		else
		{
			return {*m_curr_path / std::filesystem::path(m_file_buff.data())};
		}
	}
}
