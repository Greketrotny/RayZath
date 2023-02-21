#include "message_box.hpp"

#include "imgui.h"

namespace RayZath::UI::Windows
{
	MessageBox::MessageBox()
		: m_opened(false)
	{}
	MessageBox::MessageBox(std::string message, std::vector<std::string> options, callback_t callback)
		: m_message(std::move(message))
		, m_options(std::move(options))
		, m_callback(std::move(callback))
	{}
	MessageBox::option_t MessageBox::render()
	{
		if (!m_opened)
			return {};

		static constexpr auto popup_id = "Message##message";
		ImGui::OpenPopup(popup_id);
		ImGui::SetNextWindowSize(ImVec2(ImGui::GetWindowSize().x, -1.0f));
		ImGui::SetNextWindowPos(
			ImGui::GetWindowViewport()->GetCenter(),
			ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		std::optional<std::string> ret_opt{std::nullopt};
		if (ImGui::BeginPopupModal(popup_id, &m_opened))
		{
			Complete complete_error_popup([] { ImGui::EndPopup(); });
			ImGui::TextWrapped("%s", m_message.c_str());

			for (const auto& option : m_options)
			{
				if (ImGui::Button((option + "##MessageBoxOption").c_str()))
					ret_opt = option;
				ImGui::SameLine();
			}

			if (m_callback && ret_opt)
				m_callback(ret_opt);
		}
		if (!m_opened) ret_opt = std::string{};
		if (ret_opt) m_opened = false;
		return ret_opt;
	}
}
