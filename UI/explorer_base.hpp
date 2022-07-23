#pragma once

#include <array>
#include <string>

#include "rayzath.h"
#include "imgui.h"

namespace RayZath::UI::Windows
{
	class Filter
	{
	private:
		std::array<char, 1024> m_buffer{};
		bool m_match_case = false, m_match_word = false;

	public:
		void update()
		{
			ImGui::BeginTable("filter_table", 1, ImGuiTableFlags_BordersOuter);

			ImGui::TableNextRow();
			ImGui::TableNextColumn();

			ImGui::SetNextItemWidth(-1);
			ImGui::InputTextWithHint("##filter_input", "filter", m_buffer.data(), m_buffer.size());
			ImGui::Checkbox("match case", &m_match_case);

			ImGui::EndTable();
		}
		bool matches(const std::string& name)
		{
			std::string search_input(m_buffer.data());

			static_assert(std::tuple_size_v<decltype(m_buffer)> > 0);
			if (m_buffer[0] == '\0') return true; // no input, matches all

			if (m_match_case)
			{
				return name.find(search_input) != std::string::npos;
			}
			else
			{
				std::transform(search_input.begin(), search_input.end(), search_input.begin(),
					[](auto c) { return std::tolower(c); });
				std::string lowered_name = name;
				std::transform(lowered_name.begin(), lowered_name.end(), lowered_name.begin(),
					[](auto c) { return std::tolower(c); });
				return lowered_name.find(search_input) != std::string::npos;
			}
		}
	};

	template <RayZath::Engine::World::ObjectType T>
	class Search
	{
	public:
		using object_t = RayZath::Engine::World::object_t<T>;
	private:
		Filter m_name_filter;
		bool m_opened = true;

		RayZath::Engine::Handle<object_t> m_selected;

	public:
		std::tuple<bool, RayZath::Engine::Handle<object_t>> update(const RayZath::Engine::World& world)
		{
			static constexpr auto* popup_id = "search##search_popup_modal";
			ImGui::OpenPopup(popup_id);
			if (ImGui::BeginPopupModal(popup_id, &m_opened))
			{
				m_name_filter.update();

				if (ImGui::BeginListBox("search_list_box", ImVec2(-1.f, -1.f)))
				{
					auto& objects = world.Container<T>();
					for (uint32_t i = 0; i < objects.GetCount(); i++)
					{
						const auto& object = objects[i];
						if (!object) continue;
						if (!m_name_filter.matches(object->GetName())) continue;

						bool is_selected = m_selected == object;
						if (ImGui::Selectable(
							(object->GetName() + "##search_list_selectable" + std::to_string(i)).c_str(),
							&is_selected))
							m_selected = object;

						if (is_selected)
							ImGui::SetItemDefaultFocus();
					}
					ImGui::EndListBox();
				}
				ImGui::EndPopup();
			}

			return std::make_tuple(m_opened, m_selected);
		}
	};


	class ExplorerSelectable
	{
	public:
		struct Action
		{
			bool selected = false;
			bool double_clicked = false;
			bool right_clicked = false;
		};

		Action drawSelectable(const std::string& caption, const bool selected);
	};
	class ExplorerEditable : public ExplorerSelectable
	{
	private:
		std::array<char, 256> m_name_buffer;

	public:
		struct Action : ExplorerSelectable::Action
		{
			bool name_edited = false;

			Action() = default;
			Action(const ExplorerSelectable::Action& other_base)
				: ExplorerSelectable::Action(other_base)
			{}
		};

		Action drawEditable(const std::string& caption, const bool selected, const bool edited);
		void setNameToEdit(const std::string& name);
		std::string getEditedName();
	};
}
