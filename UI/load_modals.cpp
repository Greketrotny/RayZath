#include "load_modals.hpp"

#include "imgui.h"

namespace RayZath::UI::Windows
{
	using ObjectType = RayZath::Engine::ObjectType;

	void LoadModalBase::updateFileBrowsing()
	{
		if (ImGui::Button("browse"))
		{
			m_file_browser = FileBrowserModal{std::filesystem::current_path(), FileBrowserModal::Mode::Open};
		}
		ImGui::SameLine();
		{
			auto files = m_files_to_load.empty() ?
				std::string("<not selected>") :
				std::accumulate(
					m_files_to_load.begin(), std::prev(m_files_to_load.end()), std::string{},
					[](std::string acc, const auto& file) {
						return acc += "\"" + file.filename().string() + "\", ";
					}) + "\"" + std::prev(m_files_to_load.end())->filename().string() + "\"";
					ImGui::Text("%s", files.c_str());
		}
		if (m_file_browser)
		{
			if (m_file_browser->render())
			{
				m_files_to_load = m_file_browser->selectedFiles();
				m_file_browser.reset();
			}
		}
	}


	void LoadModal<ObjectType::Texture>::update(Scene& scene)
	{
		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
		ImGui::SetNextWindowSizeConstraints(ImVec2(300.0f, 100.0f), ImVec2(FLT_MAX, FLT_MAX));

		static constinit auto* popup_id = "load texture##load_texture_modal_window";
		if (m_opened) ImGui::OpenPopup(popup_id);
		if (ImGui::BeginPopupModal(popup_id, &m_opened))
		{
			const auto width = 300.0f;

			updateFileBrowsing();
		
			// filter mode
			ImGui::SetNextItemWidth(width);
			if (ImGui::BeginCombo(
				"filter mode##filter_listbox",
				ms_filter_modes[m_filter_mode_idx].second.data(),
				ImGuiComboFlags_HeightRegular))
			{
				for (size_t i = 0; i < ms_filter_modes.size(); i++)
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
				for (size_t i = 0; i < ms_address_modes.size(); i++)
				{
					const auto& [mode, name] = ms_address_modes[i];
					if (ImGui::Selectable(name.data()))
						m_addres_mode_idx = i;
				}
				ImGui::EndCombo();
			}
			// is hdr
			ImGui::Checkbox("HDR image", &m_is_hdr);

			ImGui::SetNextItemWidth(-1.0f);
			if (ImGui::Button("load", ImVec2(50, 0)))
			{
				for (const auto& path : m_files_to_load)
					doLoad(scene, path);
			}

			if (m_fail_message)
			{
				ImGui::SetNextItemWidth(width);
				ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 64, 64, 255));
				ImGui::TextWrapped(
					"%s",
					("Failed to load texture at specified path. Reason: " + *m_fail_message).c_str());
				ImGui::PopStyleColor();
			}
			ImGui::EndPopup();
		}
	}
	void LoadModal<ObjectType::Texture>::doLoad(Scene& scene, const std::filesystem::path& file)
	{
		try
		{
			if (m_is_hdr)
			{
				auto [texture, emission] = scene.mr_world.loader().loadHDR(file.string());
				scene.mr_world.container<Engine::ObjectType::Texture>().create(
					RZ::ConStruct<RZ::Texture>("loaded hdr rgb texture",
						std::move(texture),
						ms_filter_modes[m_filter_mode_idx].first,
						ms_address_modes[m_addres_mode_idx].first));
				scene.mr_world.container<Engine::ObjectType::EmissionMap>().create(
					RZ::ConStruct<RZ::EmissionMap>("loaded hdr emission map",
						std::move(emission),
						ms_filter_modes[m_filter_mode_idx].first,
						ms_address_modes[m_addres_mode_idx].first));
			}
			else
			{
				scene.mr_world.container<Engine::ObjectType::Texture>().create(
					RZ::ConStruct<RZ::Texture>("loaded texture",
						scene.mr_world.loader().loadMap<Engine::ObjectType::Texture>(file.string()),
						ms_filter_modes[m_filter_mode_idx].first,
						ms_address_modes[m_addres_mode_idx].first));
			}
			ImGui::CloseCurrentPopup();
			m_opened = false;
		}
		catch (std::exception& e)
		{
			if (!m_fail_message)
				m_fail_message = e.what();
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

			updateFileBrowsing();

			// filter mode
			ImGui::SetNextItemWidth(width);
			if (ImGui::BeginCombo(
				"filter mode##filter_listbox",
				ms_filter_modes[m_filter_mode_idx].second.data(),
				ImGuiComboFlags_HeightRegular))
			{
				for (size_t i = 0; i < ms_filter_modes.size(); i++)
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
				for (size_t i = 0; i < ms_address_modes.size(); i++)
				{
					const auto& [mode, name] = ms_address_modes[i];
					if (ImGui::Selectable(name.data()))
						m_addres_mode_idx = i;
				}
				ImGui::EndCombo();
			}

			// flip y axis
			ImGui::Checkbox("flip y axis###flip_y", &m_flip_y_axis);

			ImGui::SetNextItemWidth(-1.0f);
			if (ImGui::Button("load", ImVec2(50, 0)))
			{
				if (m_files_to_load.empty())
					m_fail_message = "No file selected.";
				else
				{
					for (const auto& file : m_files_to_load)
						doLoad(scene, file);
				}
			}

			if (m_fail_message)
			{
				ImGui::SetNextItemWidth(width);
				ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 64, 64, 255));
				ImGui::TextWrapped(
					"%s",
					("Failed to load normal map at specified path. Reason: " + *m_fail_message).c_str());
				ImGui::PopStyleColor();
			}
			ImGui::EndPopup();
		}
	}
	void LoadModal<ObjectType::NormalMap>::doLoad(Scene& scene, const std::filesystem::path& file)
	{
		try
		{
			auto normal_map{scene.mr_world.container<Engine::ObjectType::NormalMap>().create(
				RZ::ConStruct<RZ::NormalMap>("loaded normal map",
					scene.mr_world.loader().loadMap<Engine::ObjectType::NormalMap>(file.string()),
					ms_filter_modes[m_filter_mode_idx].first,
					ms_address_modes[m_addres_mode_idx].first))};

			if (m_flip_y_axis)
			{
				auto& bitmap = normal_map->bitmap();
				for (size_t y = 0; y < bitmap.GetHeight(); y++)
				{
					for (size_t x = 0; x < bitmap.GetWidth(); x++)
					{
						auto& value = bitmap.Value(x, y);
						value.green = 255u - value.green;
					}
				}
			}

			ImGui::CloseCurrentPopup();
			m_opened = false;
		}
		catch (std::exception& e)
		{
			if (!m_fail_message)
				m_fail_message = e.what();
		}
	}

	template <>
	void LoadModal<ObjectType::MetalnessMap>::update(Scene& scene)
	{
		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constinit auto* popup_id = "load metalness map##load_metalness_map_modal_window";
		if (m_opened) ImGui::OpenPopup(popup_id);
		if (ImGui::BeginPopupModal(popup_id, &m_opened, ImGuiWindowFlags_AlwaysAutoResize))
		{
			const auto width = 300.0f;

			updateFileBrowsing();

			// filter mode
			ImGui::SetNextItemWidth(width);
			if (ImGui::BeginCombo(
				"filter mode##filter_listbox",
				ms_filter_modes[m_filter_mode_idx].second.data(),
				ImGuiComboFlags_HeightRegular))
			{
				for (size_t i = 0; i < ms_filter_modes.size(); i++)
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
				for (size_t i = 0; i < ms_address_modes.size(); i++)
				{
					const auto& [mode, name] = ms_address_modes[i];
					if (ImGui::Selectable(name.data()))
						m_addres_mode_idx = i;
				}
				ImGui::EndCombo();
			}

			ImGui::SetNextItemWidth(-1.0f);
			if (ImGui::Button("load", ImVec2(50, 0)))
			{
				if (m_files_to_load.empty())
					m_fail_message = "No file selected.";
				else
				{
					for (const auto& file : m_files_to_load)
						doLoad(scene, file);
				}
			}

			if (m_fail_message)
			{
				ImGui::SetNextItemWidth(width);
				ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 64, 64, 255));
				ImGui::TextWrapped(
					"%s",
					("Failed to load metalness map at specified path. Reason: " + *m_fail_message).c_str());
				ImGui::PopStyleColor();
			}
			ImGui::EndPopup();
		}
	}
	template <>
	void LoadModal<ObjectType::MetalnessMap>::doLoad(Scene& scene, const std::filesystem::path& file)
	{
		try
		{
			scene.mr_world.container<Engine::ObjectType::MetalnessMap>().create(
				RZ::ConStruct<RZ::MetalnessMap>("loaded metalness map",
					scene.mr_world.loader().loadMap<Engine::ObjectType::MetalnessMap>(file.string()),
					ms_filter_modes[m_filter_mode_idx].first,
					ms_address_modes[m_addres_mode_idx].first));
			ImGui::CloseCurrentPopup();
			m_opened = false;
		}
		catch (std::exception& e)
		{
			if (!m_fail_message)
				m_fail_message = e.what();
		}
	}

	template <>
	void LoadModal<ObjectType::RoughnessMap>::update(Scene& scene)
	{
		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constinit auto* popup_id = "load roughness map##load_roughness_map_modal_window";
		if (m_opened) ImGui::OpenPopup(popup_id);
		if (ImGui::BeginPopupModal(popup_id, &m_opened, ImGuiWindowFlags_AlwaysAutoResize))
		{
			const auto width = 300.0f;

			updateFileBrowsing();

			// filter mode
			ImGui::SetNextItemWidth(width);
			if (ImGui::BeginCombo(
				"filter mode##filter_listbox",
				ms_filter_modes[m_filter_mode_idx].second.data(),
				ImGuiComboFlags_HeightRegular))
			{
				for (size_t i = 0; i < ms_filter_modes.size(); i++)
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
				for (size_t i = 0; i < ms_address_modes.size(); i++)
				{
					const auto& [mode, name] = ms_address_modes[i];
					if (ImGui::Selectable(name.data()))
						m_addres_mode_idx = i;
				}
				ImGui::EndCombo();
			}

			ImGui::SetNextItemWidth(-1.0f);
			if (ImGui::Button("load", ImVec2(50, 0)))
			{
				if (m_files_to_load.empty())
					m_fail_message = "No file selected.";
				else
				{
					for (const auto& file : m_files_to_load)
						doLoad(scene, file);
				}
			}

			if (m_fail_message)
			{
				ImGui::SetNextItemWidth(width);
				ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 64, 64, 255));
				ImGui::TextWrapped(
					"%s",
					("Failed to load roughness map at specified path. Reason: " + *m_fail_message).c_str());
				ImGui::PopStyleColor();
			}
			ImGui::EndPopup();
		}
	}
	template <>
	void LoadModal<ObjectType::RoughnessMap>::doLoad(Scene& scene, const std::filesystem::path& file)
	{
		try
		{
			scene.mr_world.container<Engine::ObjectType::RoughnessMap>().create(
				RZ::ConStruct<RZ::RoughnessMap>("loaded roughness map",
					scene.mr_world.loader().loadMap<Engine::ObjectType::RoughnessMap>(file.string()),
					ms_filter_modes[m_filter_mode_idx].first,
					ms_address_modes[m_addres_mode_idx].first));
			ImGui::CloseCurrentPopup();
			m_opened = false;
		}
		catch (std::exception& e)
		{
			if (!m_fail_message)
				m_fail_message = e.what();
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

			updateFileBrowsing();

			// filter mode
			ImGui::SetNextItemWidth(width);
			if (ImGui::BeginCombo(
				"filter mode##filter_listbox",
				ms_filter_modes[m_filter_mode_idx].second.data(),
				ImGuiComboFlags_HeightRegular))
			{
				for (size_t i = 0; i < ms_filter_modes.size(); i++)
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
				for (size_t i = 0; i < ms_address_modes.size(); i++)
				{
					const auto& [mode, name] = ms_address_modes[i];
					if (ImGui::Selectable(name.data()))
						m_addres_mode_idx = i;
				}
				ImGui::EndCombo();
			}

			ImGui::DragFloat("emission factor", &m_emission_factor, 1.0f, 0.0f, 10.0f, "%.3f");

			ImGui::SetNextItemWidth(-1.0f);
			if (ImGui::Button("load", ImVec2(50, 0)))
			{
				if (m_files_to_load.empty())
					m_fail_message = "No file selected.";
				else
				{
					for (const auto& file : m_files_to_load)
						doLoad(scene, file);
				}
			}

			if (m_fail_message)
			{
				ImGui::SetNextItemWidth(width);
				ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 64, 64, 255));
				ImGui::TextWrapped(
					"%s",
					("Failed to load roughness map at specified path. Reason: " + *m_fail_message).c_str());
				ImGui::PopStyleColor();
			}
			ImGui::EndPopup();
		}
	}
	void LoadModal<ObjectType::EmissionMap>::doLoad(Scene& scene, const std::filesystem::path& file)
	{
		try
		{
			scene.mr_world.container<Engine::ObjectType::EmissionMap>().create(
				RZ::ConStruct<RZ::EmissionMap>("loaded emission map",
					scene.mr_world.loader().loadMap<Engine::ObjectType::EmissionMap>(file.string()),
					ms_filter_modes[m_filter_mode_idx].first,
					ms_address_modes[m_addres_mode_idx].first));
			ImGui::CloseCurrentPopup();
			m_opened = false;
		}
		catch (std::exception& e)
		{
			if (!m_fail_message)
				m_fail_message = e.what();
		}
	}


	void LoadModal<ObjectType::Material>::update(Scene& scene)
	{
		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constinit auto* popup_id = "load material##load_material_modal_window";
		if (m_opened) ImGui::OpenPopup(popup_id);
		if (ImGui::BeginPopupModal(popup_id, &m_opened, ImGuiWindowFlags_AlwaysAutoResize))
		{
			const auto width = 300.0f;

			updateFileBrowsing();

			ImGui::SetNextItemWidth(-1.0f);
			if (ImGui::Button("load", ImVec2(50, 0)))
			{
				if (m_files_to_load.empty())
					m_fail_message = "No file selected.";
				else
				{
					for (const auto& file : m_files_to_load)
						doLoad(scene, file);
				}
			}

			if (m_fail_message)
			{
				ImGui::SetNextItemWidth(width);
				ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 64, 64, 255));
				ImGui::TextWrapped(
					"%s",
					("Failed to load material at specified path. Reason: " + *m_fail_message).c_str());
				ImGui::PopStyleColor();
			}
			ImGui::EndPopup();
		}
	}
	void LoadModal<ObjectType::Material>::doLoad(Scene& scene, const std::filesystem::path& file)
	{
		try
		{
			scene.mr_world.loader().loadMTL(file.string());
			ImGui::CloseCurrentPopup();
			m_opened = false;
		}
		catch (std::exception& e)
		{
			if (!m_fail_message)
				m_fail_message = e.what();
		}
	}


	void LoadModal<ObjectType::Mesh>::update(Scene& scene)
	{
		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constinit auto* popup_id = "load mesh##load_mesh_modal_window";
		if (m_opened) ImGui::OpenPopup(popup_id);
		if (ImGui::BeginPopupModal(popup_id, &m_opened, ImGuiWindowFlags_AlwaysAutoResize))
		{
			const auto width = 300.0f;

			updateFileBrowsing();

			ImGui::SetNextItemWidth(-1.0f);
			if (ImGui::Button("load", ImVec2(50, 0)))
			{
				if (m_files_to_load.empty())
					m_fail_message = "No file selected.";
				else
				{
					for (const auto& file : m_files_to_load)
						doLoad(scene, file);
				}
			}

			if (m_fail_message)
			{
				ImGui::SetNextItemWidth(width);
				ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 64, 64, 255));
				ImGui::TextWrapped(
					"%s",
					("Failed to load mesh at specified path. Reason: " + *m_fail_message).c_str());
				ImGui::PopStyleColor();
			}
			ImGui::EndPopup();
		}
	}
	void LoadModal<ObjectType::Mesh>::doLoad(Scene& scene, const std::filesystem::path& file)
	{
		try
		{
			scene.mr_world.loader().loadModel(file.string());
			ImGui::CloseCurrentPopup();
			m_opened = false;
		}
		catch (std::exception& e)
		{
			if (!m_fail_message)
				m_fail_message = e.what();
		}
	}


	void SceneLoadModal::update(Scene& scene)
	{
		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constinit auto* popup_id = "load scene##load_scene_modal_window";
		if (m_opened) ImGui::OpenPopup(popup_id);
		if (ImGui::BeginPopupModal(popup_id, &m_opened, ImGuiWindowFlags_AlwaysAutoResize))
		{
			const auto width = 300.0f;

			updateFileBrowsing();

			ImGui::SetNextItemWidth(-1.0f);
			if (ImGui::Button("load", ImVec2(50, 0)))
			{
				if (m_files_to_load.empty())
					m_fail_message = "No file selected.";
				else
				{
					for (const auto& file : m_files_to_load)
						doLoad(scene, file);
				}
			}

			if (m_fail_message)
			{
				ImGui::SetNextItemWidth(width);
				ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 64, 64, 255));
				ImGui::TextWrapped(
					"%s",
					("Failed to load scene at specified path. Reason: " + *m_fail_message).c_str());
				ImGui::PopStyleColor();
			}
			ImGui::EndPopup();
		}
	}
	void SceneLoadModal::doLoad(Scene& scene, const std::filesystem::path& file)
	{
		try
		{
			scene.mr_world.loader().loadScene(file);
			ImGui::CloseCurrentPopup();
			m_opened = false;
		}
		catch (std::exception& e)
		{
			if (!m_fail_message)
				m_fail_message = e.what();
		}
	}
}
