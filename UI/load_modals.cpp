#include "load_modals.hpp"

#include "imgui.h"

namespace RayZath::UI::Windows
{
	using ObjectType = RayZath::Engine::World::ObjectType;

	void LoadModal<ObjectType::Texture>::update(Scene& scene)
	{
		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constinit auto* popup_id = "load texture##load_texture_modal_window";
		if (m_opened) ImGui::OpenPopup(popup_id);
		if (ImGui::BeginPopupModal(popup_id, &m_opened, ImGuiWindowFlags_AlwaysAutoResize))
		{
			const auto width = 300.0f;
			// path
			ImGui::SetNextItemWidth(-1.f);
			const bool completed = ImGui::InputTextWithHint("##object_name_input", "name",
				m_path_buffer.data(), m_path_buffer.size(),
				ImGuiInputTextFlags_AlwaysOverwrite | ImGuiInputTextFlags_EnterReturnsTrue);

			// filter mode
			ImGui::SetNextItemWidth(width);
			if (ImGui::BeginCombo(
				"filter mode##filter_listbox",
				ms_filter_modes[m_filter_mode_idx].second.data(),
				ImGuiComboFlags_HeightRegular))
			{
				for (int i = 0; i < ms_filter_modes.size(); i++)
				{
					const auto& [mode, name] = ms_filter_modes[i];
					if (ImGui::Selectable(name.data()))
						m_filter_mode_idx = i;
				}
				ImGui::EndCombo();
			}
			// adress mode
			ImGui::SetNextItemWidth(width);
			if (ImGui::BeginCombo(
				"address mode##address_listbox",
				ms_address_modes[m_addres_mode_idx].second.data(),
				ImGuiComboFlags_HeightRegular))
			{
				for (int i = 0; i < ms_address_modes.size(); i++)
				{
					const auto& [mode, name] = ms_address_modes[i];
					if (ImGui::Selectable(name.data()))
						m_addres_mode_idx = i;
				}
				ImGui::EndCombo();
			}

			ImGui::SetNextItemWidth(-1.0f);
			if (ImGui::Button("load", ImVec2(50, 0)) || completed)
			{
				try
				{
					scene.mr_world.Container<Engine::World::ObjectType::Texture>().Create(
						RZ::ConStruct<RZ::Texture>("loaded texture",
							scene.mr_world.GetLoader().LoadTexture(std::string(m_path_buffer.data())),
							ms_filter_modes[m_filter_mode_idx].first,
							ms_address_modes[m_addres_mode_idx].first));
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
				ImGui::TextWrapped(("Failed to load texture at specified path. Reason: " + *m_fail_message).c_str());
				ImGui::PopStyleColor();
			}
			ImGui::EndPopup();
		}
	}
	void LoadModal<ObjectType::NormalMap>::update(Scene& scene)
	{
		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constinit auto* popup_id = "load normal map##load_normal_map_modal_window";
		if (m_opened) ImGui::OpenPopup(popup_id);
		if (ImGui::BeginPopupModal(popup_id, &m_opened, ImGuiWindowFlags_AlwaysAutoResize))
		{
			const auto width = 300.0f;
			// path
			ImGui::SetNextItemWidth(-1.f);
			const bool completed = ImGui::InputTextWithHint("##object_name_input", "name",
				m_path_buffer.data(), m_path_buffer.size(),
				ImGuiInputTextFlags_AlwaysOverwrite | ImGuiInputTextFlags_EnterReturnsTrue);

			// filter mode
			ImGui::SetNextItemWidth(width);
			if (ImGui::BeginCombo(
				"filter mode##filter_listbox",
				ms_filter_modes[m_filter_mode_idx].second.data(),
				ImGuiComboFlags_HeightRegular))
			{
				for (int i = 0; i < ms_filter_modes.size(); i++)
				{
					const auto& [mode, name] = ms_filter_modes[i];
					if (ImGui::Selectable(name.data()))
						m_filter_mode_idx = i;
				}
				ImGui::EndCombo();
			}
			// adress mode
			ImGui::SetNextItemWidth(width);
			if (ImGui::BeginCombo(
				"address mode##address_listbox",
				ms_address_modes[m_addres_mode_idx].second.data(),
				ImGuiComboFlags_HeightRegular))
			{
				for (int i = 0; i < ms_address_modes.size(); i++)
				{
					const auto& [mode, name] = ms_address_modes[i];
					if (ImGui::Selectable(name.data()))
						m_addres_mode_idx = i;
				}
				ImGui::EndCombo();
			}

			ImGui::SetNextItemWidth(-1.0f);
			if (ImGui::Button("load", ImVec2(50, 0)) || completed)
			{
				try
				{
					scene.mr_world.Container<Engine::World::ObjectType::NormalMap>().Create(
						RZ::ConStruct<RZ::NormalMap>("loaded normal map",
							scene.mr_world.GetLoader().LoadNormalMap(std::string(m_path_buffer.data())),
							ms_filter_modes[m_filter_mode_idx].first,
							ms_address_modes[m_addres_mode_idx].first));
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
				ImGui::TextWrapped(("Failed to load normal map at specified path. Reason: " + *m_fail_message).c_str());
				ImGui::PopStyleColor();
			}
			ImGui::EndPopup();
		}
	}
	void LoadModal<ObjectType::MetalnessMap>::update(Scene& scene)
	{
		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constinit auto* popup_id = "load metalness map##load_metalness_map_modal_window";
		if (m_opened) ImGui::OpenPopup(popup_id);
		if (ImGui::BeginPopupModal(popup_id, &m_opened, ImGuiWindowFlags_AlwaysAutoResize))
		{
			const auto width = 300.0f;
			// path
			ImGui::SetNextItemWidth(-1.f);
			const bool completed = ImGui::InputTextWithHint("##object_name_input", "name",
				m_path_buffer.data(), m_path_buffer.size(),
				ImGuiInputTextFlags_AlwaysOverwrite | ImGuiInputTextFlags_EnterReturnsTrue);

			// filter mode
			ImGui::SetNextItemWidth(width);
			if (ImGui::BeginCombo(
				"filter mode##filter_listbox",
				ms_filter_modes[m_filter_mode_idx].second.data(),
				ImGuiComboFlags_HeightRegular))
			{
				for (int i = 0; i < ms_filter_modes.size(); i++)
				{
					const auto& [mode, name] = ms_filter_modes[i];
					if (ImGui::Selectable(name.data()))
						m_filter_mode_idx = i;
				}
				ImGui::EndCombo();
			}
			// adress mode
			ImGui::SetNextItemWidth(width);
			if (ImGui::BeginCombo(
				"address mode##address_listbox",
				ms_address_modes[m_addres_mode_idx].second.data(),
				ImGuiComboFlags_HeightRegular))
			{
				for (int i = 0; i < ms_address_modes.size(); i++)
				{
					const auto& [mode, name] = ms_address_modes[i];
					if (ImGui::Selectable(name.data()))
						m_addres_mode_idx = i;
				}
				ImGui::EndCombo();
			}

			ImGui::SetNextItemWidth(-1.0f);
			if (ImGui::Button("load", ImVec2(50, 0)) || completed)
			{
				try
				{
					scene.mr_world.Container<Engine::World::ObjectType::MetalnessMap>().Create(
						RZ::ConStruct<RZ::MetalnessMap>("loaded metalness map",
							scene.mr_world.GetLoader().LoadMetalnessMap(std::string(m_path_buffer.data())),
							ms_filter_modes[m_filter_mode_idx].first,
							ms_address_modes[m_addres_mode_idx].first));
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
				ImGui::TextWrapped(("Failed to load metalness map at specified path. Reason: " + *m_fail_message).c_str());
				ImGui::PopStyleColor();
			}
			ImGui::EndPopup();
		}
	}
	void LoadModal<ObjectType::RoughnessMap>::update(Scene& scene)
	{
		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constinit auto* popup_id = "load roughness map##load_roughness_map_modal_window";
		if (m_opened) ImGui::OpenPopup(popup_id);
		if (ImGui::BeginPopupModal(popup_id, &m_opened, ImGuiWindowFlags_AlwaysAutoResize))
		{
			const auto width = 300.0f;
			// path
			ImGui::SetNextItemWidth(-1.f);
			const bool completed = ImGui::InputTextWithHint("##object_name_input", "name",
				m_path_buffer.data(), m_path_buffer.size(),
				ImGuiInputTextFlags_AlwaysOverwrite | ImGuiInputTextFlags_EnterReturnsTrue);

			// filter mode
			ImGui::SetNextItemWidth(width);
			if (ImGui::BeginCombo(
				"filter mode##filter_listbox",
				ms_filter_modes[m_filter_mode_idx].second.data(),
				ImGuiComboFlags_HeightRegular))
			{
				for (int i = 0; i < ms_filter_modes.size(); i++)
				{
					const auto& [mode, name] = ms_filter_modes[i];
					if (ImGui::Selectable(name.data()))
						m_filter_mode_idx = i;
				}
				ImGui::EndCombo();
			}
			// adress mode
			ImGui::SetNextItemWidth(width);
			if (ImGui::BeginCombo(
				"address mode##address_listbox",
				ms_address_modes[m_addres_mode_idx].second.data(),
				ImGuiComboFlags_HeightRegular))
			{
				for (int i = 0; i < ms_address_modes.size(); i++)
				{
					const auto& [mode, name] = ms_address_modes[i];
					if (ImGui::Selectable(name.data()))
						m_addres_mode_idx = i;
				}
				ImGui::EndCombo();
			}

			ImGui::SetNextItemWidth(-1.0f);
			if (ImGui::Button("load", ImVec2(50, 0)) || completed)
			{
				try
				{
					scene.mr_world.Container<Engine::World::ObjectType::RoughnessMap>().Create(
						RZ::ConStruct<RZ::RoughnessMap>("loaded roughness map",
							scene.mr_world.GetLoader().LoadRoughnessMap(std::string(m_path_buffer.data())),
							ms_filter_modes[m_filter_mode_idx].first,
							ms_address_modes[m_addres_mode_idx].first));
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
				ImGui::TextWrapped(("Failed to load roughness map at specified path. Reason: " + *m_fail_message).c_str());
				ImGui::PopStyleColor();
			}
			ImGui::EndPopup();
		}
	}
	void LoadModal<ObjectType::EmissionMap>::update(Scene& scene)
	{
		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constinit auto* popup_id = "load emission map##load_emission_map_modal_window";
		if (m_opened) ImGui::OpenPopup(popup_id);
		if (ImGui::BeginPopupModal(popup_id, &m_opened, ImGuiWindowFlags_AlwaysAutoResize))
		{
			const auto width = 300.0f;
			// path
			ImGui::SetNextItemWidth(-1.f);
			const bool completed = ImGui::InputTextWithHint("##object_name_input", "name",
				m_path_buffer.data(), m_path_buffer.size(),
				ImGuiInputTextFlags_AlwaysOverwrite | ImGuiInputTextFlags_EnterReturnsTrue);

			// filter mode
			ImGui::SetNextItemWidth(width);
			if (ImGui::BeginCombo(
				"filter mode##filter_listbox",
				ms_filter_modes[m_filter_mode_idx].second.data(),
				ImGuiComboFlags_HeightRegular))
			{
				for (int i = 0; i < ms_filter_modes.size(); i++)
				{
					const auto& [mode, name] = ms_filter_modes[i];
					if (ImGui::Selectable(name.data()))
						m_filter_mode_idx = i;
				}
				ImGui::EndCombo();
			}
			// adress mode
			ImGui::SetNextItemWidth(width);
			if (ImGui::BeginCombo(
				"address mode##address_listbox",
				ms_address_modes[m_addres_mode_idx].second.data(),
				ImGuiComboFlags_HeightRegular))
			{
				for (int i = 0; i < ms_address_modes.size(); i++)
				{
					const auto& [mode, name] = ms_address_modes[i];
					if (ImGui::Selectable(name.data()))
						m_addres_mode_idx = i;
				}
				ImGui::EndCombo();
			}

			ImGui::DragFloat("emission factor", &m_emission_factor, 1.0f, 0.0f, 10.0f, "%.3f");

			ImGui::SetNextItemWidth(-1.0f);
			if (ImGui::Button("load", ImVec2(50, 0)) || completed)
			{
				try
				{
					scene.mr_world.Container<Engine::World::ObjectType::EmissionMap>().Create(
						RZ::ConStruct<RZ::EmissionMap>("loaded emission map",
							scene.mr_world.GetLoader().LoadEmissionMap(std::string(m_path_buffer.data())),
							ms_filter_modes[m_filter_mode_idx].first,
							ms_address_modes[m_addres_mode_idx].first));
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
				ImGui::TextWrapped(("Failed to load roughness map at specified path. Reason: " + *m_fail_message).c_str());
				ImGui::PopStyleColor();
			}
			ImGui::EndPopup();
		}
	}
}
