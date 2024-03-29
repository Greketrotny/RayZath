#include "save_modals.hpp"

#include "imgui.h"

namespace RayZath::UI::Windows
{
	void SaveModalBase::updateFileBrowsing()
	{
		if (ImGui::Button("browse"))
		{
			m_file_browser = FileBrowserModal{
				m_file_to_save.empty() || std::filesystem::exists(m_file_to_save) ? 
				std::filesystem::current_path() : m_file_to_save.parent_path(),
				FileBrowserModal::Mode::Save};
		}
		ImGui::SameLine();
		const auto file = m_file_to_save.empty() ?
			std::string("<path not selected>") :
			m_file_to_save.string();
		ImGui::Text("%s", file.c_str());

		if (m_file_browser)
		{
			if (m_file_browser->render())
			{
				auto selected = m_file_browser->selectedFiles();
				RZAssertCore(!selected.empty(), "file browser has files on success.");
				m_file_to_save = selected[0];
				m_file_browser.reset();
			}
		}

	}

	template<> void MapSaveModal<Engine::ObjectType::Texture>::update(Scene& scene)
	{
		if (!m_opened) return;

		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constexpr auto* popup_id = "save texture##save_texture_modal_window";
		ImGui::OpenPopup(popup_id);
		if (!ImGui::BeginPopupModal(popup_id, &m_opened))
			return;

		// map selection modal
		if (ImGui::Button("Select"))
			m_search_modal = std::make_unique<Search<Engine::ObjectType::Texture>>();
		if (m_search_modal)
		{
			if (const auto [opened, map] = m_search_modal->update(scene.mr_world); !opened || map)
			{
				m_selected_map = map;
				m_search_modal.reset();
			}
		}
		ImGui::SameLine();
		ImGui::Text("%s", m_selected_map ? m_selected_map->name().c_str() : "<texture to save>");

		updateFileBrowsing();

		ImGui::SetNextItemWidth(-1.0f);
		if ((ImGui::Button("save", ImVec2(50, 0))) && m_selected_map)
		{
			try
			{
				scene.mr_world.saver().saveMap<Engine::ObjectType::Texture>(
					m_selected_map->bitmap(),
					m_file_to_save,
					m_selected_map->name());
				m_opened = false;
			}
			catch (std::exception& e)
			{
				m_message_box = MessageBox(
					e.what(), {"Ok"});
			}
		}
		m_message_box.render();
		ImGui::EndPopup();
	}
	template<> void MapSaveModal<Engine::ObjectType::NormalMap>::update(Scene& scene)
	{
		if (!m_opened) return;

		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constexpr auto* popup_id = "save normal map##save_normal_map_modal_window";
		ImGui::OpenPopup(popup_id);
		if (!ImGui::BeginPopupModal(popup_id, &m_opened))
			return;

		// map selection modal
		if (ImGui::Button("Select"))
			m_search_modal = std::make_unique<Search<Engine::ObjectType::NormalMap>>();
		if (m_search_modal)
		{
			if (const auto [opened, map] = m_search_modal->update(scene.mr_world); !opened || map)
			{
				m_selected_map = map;
				m_search_modal.reset();
			}
		}
		ImGui::SameLine();
		ImGui::Text("%s", m_selected_map ? m_selected_map->name().c_str() : "<normal map to save>");

		updateFileBrowsing();

		ImGui::SetNextItemWidth(-1.0f);
		if ((ImGui::Button("save", ImVec2(50, 0))) && m_selected_map)
		{
			try
			{
				scene.mr_world.saver().saveMap<Engine::ObjectType::NormalMap>(
					m_selected_map->bitmap(),
					m_file_to_save,
					m_selected_map->name());
				m_opened = false;
			}
			catch (std::exception& e)
			{
				m_message_box = MessageBox(
					e.what(), {"Ok"});
			}
		}
		m_message_box.render();
		ImGui::EndPopup();
	}
	template<> void MapSaveModal<Engine::ObjectType::MetalnessMap>::update(Scene& scene)
	{
		if (!m_opened) return;

		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constexpr auto* popup_id = "save metalness map##save_metalness_map_modal_window";
		ImGui::OpenPopup(popup_id);
		if (!ImGui::BeginPopupModal(popup_id, &m_opened))
			return;

		// map selection modal
		if (ImGui::Button("Select"))
			m_search_modal = std::make_unique<Search<Engine::ObjectType::MetalnessMap>>();
		if (m_search_modal)
		{
			if (const auto [opened, map] = m_search_modal->update(scene.mr_world); !opened || map)
			{
				m_selected_map = map;
				m_search_modal.reset();
			}
		}
		ImGui::SameLine();
		ImGui::Text("%s", m_selected_map ? m_selected_map->name().c_str() : "<metalness map to save>");

		updateFileBrowsing();

		ImGui::SetNextItemWidth(-1.0f);
		if ((ImGui::Button("save", ImVec2(50, 0))) && m_selected_map)
		{
			try
			{
				scene.mr_world.saver().saveMap<Engine::ObjectType::MetalnessMap>(
					m_selected_map->bitmap(),
					m_file_to_save,
					m_selected_map->name());
				m_opened = false;
			}
			catch (std::exception& e)
			{
				m_message_box = MessageBox(
					e.what(), {"Ok"});
			}
		}
		m_message_box.render();
		ImGui::EndPopup();
	}
	template<> void MapSaveModal<Engine::ObjectType::RoughnessMap>::update(Scene& scene)
	{
		if (!m_opened) return;

		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constexpr auto* popup_id = "save roughness map##save_roughness_map_modal_window";
		ImGui::OpenPopup(popup_id);
		if (!ImGui::BeginPopupModal(popup_id, &m_opened))
			return;

		// map selection modal
		if (ImGui::Button("Select"))
			m_search_modal = std::make_unique<Search<Engine::ObjectType::RoughnessMap>>();
		if (m_search_modal)
		{
			if (const auto [opened, map] = m_search_modal->update(scene.mr_world); !opened || map)
			{
				m_selected_map = map;
				m_search_modal.reset();
			}
		}
		ImGui::SameLine();
		ImGui::Text("%s", m_selected_map ? m_selected_map->name().c_str() : "<roughness map to save>");

		updateFileBrowsing();

		ImGui::SetNextItemWidth(-1.0f);
		if ((ImGui::Button("save", ImVec2(50, 0))) && m_selected_map)
		{
			try
			{
				scene.mr_world.saver().saveMap<Engine::ObjectType::RoughnessMap>(
					m_selected_map->bitmap(),
					m_file_to_save,
					m_selected_map->name());
				m_opened = false;
			}
			catch (std::exception& e)
			{
				m_message_box = MessageBox(
					e.what(), {"Ok"});
			}
		}
		m_message_box.render();
		ImGui::EndPopup();
	}
	template<> void MapSaveModal<Engine::ObjectType::EmissionMap>::update(Scene& scene)
	{
		if (!m_opened) return;

		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constexpr auto* popup_id = "save emission map##save_emission_map_modal_window";
		ImGui::OpenPopup(popup_id);
		if (!ImGui::BeginPopupModal(popup_id, &m_opened))
			return;

		// map selection modal
		if (ImGui::Button("Select"))
			m_search_modal = std::make_unique<Search<Engine::ObjectType::EmissionMap>>();
		if (m_search_modal)
		{
			if (const auto [opened, map] = m_search_modal->update(scene.mr_world); !opened || map)
			{
				m_selected_map = map;
				m_search_modal.reset();
			}
		}
		ImGui::SameLine();
		ImGui::Text("%s", m_selected_map ? m_selected_map->name().c_str() : "<emission map to save>");

		updateFileBrowsing();

		ImGui::SetNextItemWidth(-1.0f);
		if ((ImGui::Button("save", ImVec2(50, 0))) && m_selected_map)
		{
			try
			{
				scene.mr_world.saver().saveMap<Engine::ObjectType::EmissionMap>(
					m_selected_map->bitmap(),
					m_file_to_save,
					m_selected_map->name());
				m_opened = false;
			}
			catch (std::exception& e)
			{
				m_message_box = MessageBox(
					e.what(), {"Ok"});
			}
		}
		m_message_box.render();
		ImGui::EndPopup();
	}

	void MTLSaveModal::update(Scene& scene)
	{
		if (!m_opened) return;

		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constexpr auto* popup_id = "save material##save_material_modal_window";
		ImGui::OpenPopup(popup_id);
		if (!ImGui::BeginPopupModal(popup_id, &m_opened))
			return;

		if (ImGui::Button("Select"))
			m_search_modal = std::make_unique<Search<Engine::ObjectType::Material>>();
		if (m_search_modal)
		{
			if (const auto [opened, material] = m_search_modal->update(scene.mr_world); !opened || material)
			{
				m_selected_material = material;
				m_search_modal.reset();
			}
		}
		ImGui::SameLine();
		ImGui::Text("%s", m_selected_material ? m_selected_material->name().c_str() : "<material to save>");

		updateFileBrowsing();

		ImGui::SetNextItemWidth(-1.0f);
		if ((ImGui::Button("save", ImVec2(50, 0))) && m_selected_material)
		{
			try
			{
				const auto& path = m_file_to_save.has_filename() ? m_file_to_save.parent_path() : m_file_to_save;
				const auto& file_name = m_file_to_save.has_filename() ?
					m_file_to_save.filename().string() :
					m_selected_material->name();

				scene.mr_world.saver().saveMTLWithMaps(*m_selected_material, path, file_name);
				m_opened = false;
			}
			catch (std::exception& e)
			{
				m_message_box = MessageBox(
					e.what(), {"Ok"});
			}
		}
		m_message_box.render();
		ImGui::EndPopup();
	}

	void OBJSaveModal::update(Scene& scene)
	{
		if (!m_opened) return;

		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constexpr auto* popup_id = "save mesh##save_mesh_modal_window";
		ImGui::OpenPopup(popup_id);
		if (!ImGui::BeginPopupModal(popup_id, &m_opened))
			return;

		// map selection modal
		if (ImGui::Button("Select"))
			m_search_modal = std::make_unique<Search<Engine::ObjectType::Mesh>>();
		if (m_search_modal)
		{
			if (const auto [opened, mesh] = m_search_modal->update(scene.mr_world); !opened || mesh)
			{
				m_selected_mesh = mesh;
				m_search_modal.reset();
			}
		}
		ImGui::SameLine();
		ImGui::Text("%s", m_selected_mesh ? m_selected_mesh->name().c_str() : "<model to save>");

		updateFileBrowsing();

		ImGui::SetNextItemWidth(-1.0f);
		if ((ImGui::Button("save", ImVec2(50, 0))) && m_selected_mesh)
		{
			try
			{
				scene.mr_world.saver().saveOBJ(
					*m_selected_mesh,
					m_file_to_save,
					std::nullopt,
					{});
				m_opened = false;
			}
			catch (std::exception& e)
			{
				m_message_box = MessageBox(
					e.what(), {"Ok"});
			}
		}
		m_message_box.render();
		ImGui::EndPopup();
	}

	void InstanceSaveModal::update(Scene& scene)
	{
		if (!m_opened) return;

		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constexpr auto* popup_id = "save instance##save_instance_modal_window";
		ImGui::OpenPopup(popup_id);
		if (!ImGui::BeginPopupModal(popup_id, &m_opened))
			return;

		// map selection modal
		if (ImGui::Button("Select"))
			m_search_modal = std::make_unique<Search<Engine::ObjectType::Instance>>();
		if (m_search_modal)
		{
			if (const auto [opened, instance] = m_search_modal->update(scene.mr_world); !opened || instance)
			{
				m_selected_instance = instance;
				m_search_modal.reset();
			}
		}
		ImGui::SameLine();
		ImGui::Text("%s", m_selected_instance ? m_selected_instance->name().c_str() : "<instance to save>");

		updateFileBrowsing();

		ImGui::SetNextItemWidth(-1.0f);
		if ((ImGui::Button("save", ImVec2(50, 0))) && m_selected_instance)
		{
			try
			{
				scene.mr_world.saver().saveOBJ(
					{m_selected_instance},
					m_file_to_save);
				m_opened = false;
			}
			catch (std::exception& e)
			{
				m_message_box = MessageBox(
					e.what(), {"Ok"});
			}
		}
		m_message_box.render();
		ImGui::EndPopup();
	}

	void ModelSaveModal::update(Scene& scene)
	{
		if (!m_opened) return;

		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constexpr auto* popup_id = "save model##save_model_modal_window";
		ImGui::OpenPopup(popup_id);
		if (!ImGui::BeginPopupModal(popup_id, &m_opened))
			return;

		// map selection modal
		if (ImGui::Button("Select"))
			m_search_modal = std::make_unique<Search<Engine::ObjectType::Group>>();
		if (m_search_modal)
		{
			if (const auto [opened, model] = m_search_modal->update(scene.mr_world); !opened || model)
			{
				m_selected_group = model;
				m_search_modal.reset();
			}
		}
		ImGui::SameLine();
		ImGui::Text("%s", m_selected_group ? m_selected_group->name().c_str() : "<model to save>");

		updateFileBrowsing();

		ImGui::SetNextItemWidth(-1.0f);
		if ((ImGui::Button("save", ImVec2(50, 0))) && m_selected_group)
		{
			if (!m_selected_group->groups().empty())
			{
				m_message_box = MessageBox("Selected group as model can't have subgroups.", {"Ok"});
			}
			else
			{
				try
				{
					scene.mr_world.saver().saveOBJ(
						m_selected_group->objects(),
						m_file_to_save);
					m_opened = false;
				}
				catch (std::exception& e)
				{
					m_message_box = MessageBox(
						e.what(), {"Ok"});
				}
			}
		}
		m_message_box.render();
		ImGui::EndPopup();
	}

	void SceneSaveModal::update(Scene& scene)
	{
		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constexpr auto* popup_id = "save scene##save_scene_modal_window";
		if (m_opened) ImGui::OpenPopup(popup_id);
		if (ImGui::BeginPopupModal(popup_id, &m_opened, ImGuiWindowFlags_AlwaysAutoResize))
		{
			updateFileBrowsing();

			// allow partial write
			ImGui::Checkbox("allow partial write", &m_save_options.allow_partial_write);

			ImGui::SetNextItemWidth(-1.0f);
			if (ImGui::Button("save", ImVec2(50, 0)))
			{
				try
				{
					m_save_options.path = m_file_to_save;
					scene.mr_world.saver().saveScene(m_save_options);
					ImGui::CloseCurrentPopup();
					m_opened = false;
					throw std::runtime_error("saving scene failed");
				}
				catch (std::exception& e)
				{
					m_message_box = MessageBox(
						e.what(), {"Ok"});
				}
			}
			m_message_box.render();
			ImGui::EndPopup();
		}
	}
}
