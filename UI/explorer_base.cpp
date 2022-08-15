#include "explorer.hpp"

#include "imgui.h"
#include "rayzath.hpp"

#include <ranges>
#include <iostream>

namespace RZ = RayZath::Engine;

namespace RayZath::UI::Windows
{
	ExplorerSelectable::Action ExplorerSelectable::drawSelectable(const std::string& caption, const bool selected)
	{
		Action action{};
		if (ImGui::Selectable(
			caption.c_str(),
			selected,
			ImGuiSelectableFlags_AllowDoubleClick))
		{
			action.selected = true;
			if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
			{
				action.double_clicked = true;
			}
		}
		if (ImGui::IsItemClicked(ImGuiMouseButton_Right))
			action.right_clicked = true;

		return action;
	}


	ExplorerEditable::Action ExplorerEditable::drawEditable(
		const std::string& caption,
		const bool selected, const bool edited,
		const float width)
	{
		if (edited)
		{
			Action action{};
			ImGui::PushItemWidth(width);
			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
			if (ImGui::InputText(
				caption.c_str(),
				m_name_buffer.data(), sizeof(m_name_buffer) / sizeof(m_name_buffer[0]),
				ImGuiInputTextFlags_AllowTabInput |
				ImGuiInputTextFlags_AutoSelectAll |
				ImGuiInputTextFlags_EnterReturnsTrue))
			{
				action.name_edited = true;
			}
			ImGui::PopStyleVar();
			ImGui::PopItemWidth();
			return action;
		}
		else
		{
			return drawSelectable(caption, selected);
		}
	}
	void ExplorerEditable::setNameToEdit(const std::string& name)
	{
		std::memset(m_name_buffer.data(), 0, m_name_buffer.size() * sizeof(decltype(m_name_buffer)::value_type));
		std::memcpy(
			m_name_buffer.data(), name.c_str(),
			sizeof(decltype(m_name_buffer)::value_type) * std::min(m_name_buffer.size(), name.length()));
	}
	std::string ExplorerEditable::getEditedName()
	{
		return std::string(m_name_buffer.data());
	}
}
