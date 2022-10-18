#include "save_modals.hpp"

#include "imgui.h"

namespace RayZath::UI::Windows
{
	template<> void MapSaveModal<Engine::World::ObjectType::Texture>::update(Scene& scene)
	{
		if (!m_opened) return;

		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constinit auto* popup_id = "save texture##save_texture_modal_window";
		ImGui::OpenPopup(popup_id);
		if (!ImGui::BeginPopupModal(popup_id, &m_opened))
			return;

		// map selection modal
		if (ImGui::Button("Select"))
			m_search_modal = std::make_unique<Search<Engine::World::ObjectType::Texture>>();
		if (m_search_modal)
		{
			if (const auto [opened, map] = m_search_modal->update(scene.mr_world); !opened || map)
			{
				m_selected_map = map;
				m_search_modal.reset();
			}
		}
		ImGui::SameLine();
		ImGui::Text("%s", m_selected_map ? m_selected_map->name().c_str() : "<not selected>");


		const auto width = 300.0f;
		// path
		ImGui::SetNextItemWidth(-1.f);
		const bool completed = ImGui::InputTextWithHint("##object_name_input", "name",
			m_path_buffer.data(), m_path_buffer.size(), ImGuiInputTextFlags_EnterReturnsTrue);

		ImGui::SetNextItemWidth(-1.0f);
		if ((ImGui::Button("save", ImVec2(50, 0)) || completed) && m_selected_map)
		{
			try
			{
				scene.mr_world.saver().saveMap<Engine::World::ObjectType::Texture>(
					m_selected_map->bitmap(),
					std::filesystem::path(std::string(m_path_buffer.data())),
					m_selected_map->name());
				m_opened = false;
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
			ImGui::TextWrapped("%s", ("Failed to save texture at specified path. Reason: " + *m_fail_message).c_str());
			ImGui::PopStyleColor();
		}
		ImGui::EndPopup();
	}
	template<> void MapSaveModal<Engine::World::ObjectType::NormalMap>::update(Scene& scene)
	{
		if (!m_opened) return;

		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constinit auto* popup_id = "save normal map##save_normal_map_modal_window";
		ImGui::OpenPopup(popup_id);
		if (!ImGui::BeginPopupModal(popup_id, &m_opened))
			return;

		// map selection modal
		if (ImGui::Button("Select"))
			m_search_modal = std::make_unique<Search<Engine::World::ObjectType::NormalMap>>();
		if (m_search_modal)
		{
			if (const auto [opened, map] = m_search_modal->update(scene.mr_world); !opened || map)
			{
				m_selected_map = map;
				m_search_modal.reset();
			}
		}
		ImGui::SameLine();
		ImGui::Text("%s", m_selected_map ? m_selected_map->name().c_str() : "<not selected>");


		const auto width = 300.0f;
		// path
		ImGui::SetNextItemWidth(-1.f);
		const bool completed = ImGui::InputTextWithHint("##object_name_input", "name",
			m_path_buffer.data(), m_path_buffer.size(), ImGuiInputTextFlags_EnterReturnsTrue);

		ImGui::SetNextItemWidth(-1.0f);
		if ((ImGui::Button("save", ImVec2(50, 0)) || completed) && m_selected_map)
		{
			try
			{
				scene.mr_world.saver().saveMap<Engine::World::ObjectType::NormalMap>(
					m_selected_map->bitmap(),
					std::filesystem::path(std::string(m_path_buffer.data())),
					m_selected_map->name());
				m_opened = false;
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
			ImGui::TextWrapped("%s", ("Failed to save normal map at specified path. Reason: " + *m_fail_message).c_str());
			ImGui::PopStyleColor();
		}
		ImGui::EndPopup();
	}
	template<> void MapSaveModal<Engine::World::ObjectType::MetalnessMap>::update(Scene& scene)
	{
		if (!m_opened) return;

		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constinit auto* popup_id = "save metalness map##save_metalness_map_modal_window";
		ImGui::OpenPopup(popup_id);
		if (!ImGui::BeginPopupModal(popup_id, &m_opened))
			return;

		// map selection modal
		if (ImGui::Button("Select"))
			m_search_modal = std::make_unique<Search<Engine::World::ObjectType::MetalnessMap>>();
		if (m_search_modal)
		{
			if (const auto [opened, map] = m_search_modal->update(scene.mr_world); !opened || map)
			{
				m_selected_map = map;
				m_search_modal.reset();
			}
		}
		ImGui::SameLine();
		ImGui::Text("%s", m_selected_map ? m_selected_map->name().c_str() : "<not selected>");


		const auto width = 300.0f;
		// path
		ImGui::SetNextItemWidth(-1.f);
		const bool completed = ImGui::InputTextWithHint("##object_name_input", "name",
			m_path_buffer.data(), m_path_buffer.size(), ImGuiInputTextFlags_EnterReturnsTrue);

		ImGui::SetNextItemWidth(-1.0f);
		if ((ImGui::Button("save", ImVec2(50, 0)) || completed) && m_selected_map)
		{
			try
			{
				scene.mr_world.saver().saveMap<Engine::World::ObjectType::MetalnessMap>(
					m_selected_map->bitmap(),
					std::filesystem::path(std::string(m_path_buffer.data())),
					m_selected_map->name());
				m_opened = false;
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
			ImGui::TextWrapped("%s", ("Failed to save metalness map at specified path. Reason: " + *m_fail_message).c_str());
			ImGui::PopStyleColor();
		}
		ImGui::EndPopup();
	}
	template<> void MapSaveModal<Engine::World::ObjectType::RoughnessMap>::update(Scene& scene)
	{
		if (!m_opened) return;

		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constinit auto* popup_id = "save roughness map##save_roughness_map_modal_window";
		ImGui::OpenPopup(popup_id);
		if (!ImGui::BeginPopupModal(popup_id, &m_opened))
			return;

		// map selection modal
		if (ImGui::Button("Select"))
			m_search_modal = std::make_unique<Search<Engine::World::ObjectType::RoughnessMap>>();
		if (m_search_modal)
		{
			if (const auto [opened, map] = m_search_modal->update(scene.mr_world); !opened || map)
			{
				m_selected_map = map;
				m_search_modal.reset();
			}
		}
		ImGui::SameLine();
		ImGui::Text("%s", m_selected_map ? m_selected_map->name().c_str() : "<not selected>");


		const auto width = 300.0f;
		// path
		ImGui::SetNextItemWidth(-1.f);
		const bool completed = ImGui::InputTextWithHint("##object_name_input", "name",
			m_path_buffer.data(), m_path_buffer.size(), ImGuiInputTextFlags_EnterReturnsTrue);

		ImGui::SetNextItemWidth(-1.0f);
		if ((ImGui::Button("save", ImVec2(50, 0)) || completed) && m_selected_map)
		{
			try
			{
				scene.mr_world.saver().saveMap<Engine::World::ObjectType::RoughnessMap>(
					m_selected_map->bitmap(),
					std::filesystem::path(std::string(m_path_buffer.data())),
					m_selected_map->name());
				m_opened = false;
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
			ImGui::TextWrapped("%s", ("Failed to save roughness map at specified path. Reason: " + *m_fail_message).c_str());
			ImGui::PopStyleColor();
		}
		ImGui::EndPopup();
	}
	template<> void MapSaveModal<Engine::World::ObjectType::EmissionMap>::update(Scene& scene)
	{
		if (!m_opened) return;

		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constinit auto* popup_id = "save emission map##save_emission_map_modal_window";
		ImGui::OpenPopup(popup_id);
		if (!ImGui::BeginPopupModal(popup_id, &m_opened))
			return;

		// map selection modal
		if (ImGui::Button("Select"))
			m_search_modal = std::make_unique<Search<Engine::World::ObjectType::EmissionMap>>();
		if (m_search_modal)
		{
			if (const auto [opened, map] = m_search_modal->update(scene.mr_world); !opened || map)
			{
				m_selected_map = map;
				m_search_modal.reset();
			}
		}
		ImGui::SameLine();
		ImGui::Text("%s", m_selected_map ? m_selected_map->name().c_str() : "<not selected>");


		const auto width = 300.0f;
		// path
		ImGui::SetNextItemWidth(-1.f);
		const bool completed = ImGui::InputTextWithHint("##object_name_input", "name",
			m_path_buffer.data(), m_path_buffer.size(), ImGuiInputTextFlags_EnterReturnsTrue);

		ImGui::SetNextItemWidth(-1.0f);
		if ((ImGui::Button("save", ImVec2(50, 0)) || completed) && m_selected_map)
		{
			try
			{
				scene.mr_world.saver().saveMap<Engine::World::ObjectType::EmissionMap>(
					m_selected_map->bitmap(),
					std::filesystem::path(std::string(m_path_buffer.data())),
					m_selected_map->name());
				m_opened = false;
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
			ImGui::TextWrapped("%s", ("Failed to save emission map at specified path. Reason: " + *m_fail_message).c_str());
			ImGui::PopStyleColor();
		}
		ImGui::EndPopup();
	}

	void MTLSaveModal::update(Scene& scene)
	{
		if (!m_opened) return;

		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constinit auto* popup_id = "save material##save_material_modal_window";
		ImGui::OpenPopup(popup_id);
		if (!ImGui::BeginPopupModal(popup_id, &m_opened))
			return;

		if (ImGui::Button("Select"))
			m_search_modal = std::make_unique<Search<Engine::World::ObjectType::Material>>();
		if (m_search_modal)
		{
			if (const auto [opened, material] = m_search_modal->update(scene.mr_world); !opened || material)
			{
				m_selected_material = material;
				m_search_modal.reset();
			}
		}
		ImGui::SameLine();
		ImGui::Text("%s", m_selected_material ? m_selected_material->name().c_str() : "<not selected>");


		const auto width = 300.0f;
		// path
		ImGui::SetNextItemWidth(-1.f);
		const bool completed = ImGui::InputTextWithHint("##object_name_input", "name",
			m_path_buffer.data(), m_path_buffer.size(), ImGuiInputTextFlags_EnterReturnsTrue);

		ImGui::SetNextItemWidth(-1.0f);
		if ((ImGui::Button("save", ImVec2(50, 0)) || completed) && m_selected_material)
		{
			try
			{
				const auto entered_path = std::filesystem::path(std::string(m_path_buffer.data()));
				const auto& path = entered_path.has_filename() ? entered_path.parent_path() : entered_path;
				const auto& file_name = entered_path.has_filename() ? 
					entered_path.filename().string() : 
					m_selected_material->name();

				scene.mr_world.saver().saveMTLWithMaps(*m_selected_material, path, file_name);
				m_opened = false;
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
			ImGui::TextWrapped("%s", ("Failed to save material at specified path. Reason: " + *m_fail_message).c_str());
			ImGui::PopStyleColor();
		}
		ImGui::EndPopup();
	}

	void OBJSaveModal::update(Scene& scene)
	{
		if (!m_opened) return;

		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constinit auto* popup_id = "save mesh##save_mesh_modal_window";
		ImGui::OpenPopup(popup_id);
		if (!ImGui::BeginPopupModal(popup_id, &m_opened))
			return;

		// map selection modal
		if (ImGui::Button("Select"))
			m_search_modal = std::make_unique<Search<Engine::World::ObjectType::Mesh>>();
		if (m_search_modal)
		{
			if (const auto [opened, mesh] = m_search_modal->update(scene.mr_world); !opened || mesh)
			{
				m_selected_mesh = mesh;
				m_search_modal.reset();
			}
		}
		ImGui::SameLine();
		ImGui::Text("%s", m_selected_mesh ? m_selected_mesh->name().c_str() : "<not selected>");


		const auto width = 300.0f;
		// path
		ImGui::SetNextItemWidth(-1.f);
		const bool completed = ImGui::InputTextWithHint("##object_name_input", "name",
			m_path_buffer.data(), m_path_buffer.size(), ImGuiInputTextFlags_EnterReturnsTrue);

		ImGui::SetNextItemWidth(-1.0f);
		if ((ImGui::Button("save", ImVec2(50, 0)) || completed) && m_selected_mesh)
		{
			try
			{
				scene.mr_world.saver().saveOBJ(
					*m_selected_mesh,
					std::filesystem::path(std::string(m_path_buffer.data())),
					std::nullopt,
					{});
				m_opened = false;
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
			ImGui::TextWrapped("%s", ("Failed to save mesh at specified path. Reason: " + *m_fail_message).c_str());
			ImGui::PopStyleColor();
		}
		ImGui::EndPopup();
	}

	void InstanceSaveModal::update(Scene& scene)
	{
		if (!m_opened) return;

		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constinit auto* popup_id = "save instance##save_instance_modal_window";
		ImGui::OpenPopup(popup_id);
		if (!ImGui::BeginPopupModal(popup_id, &m_opened))
			return;

		// map selection modal
		if (ImGui::Button("Select"))
			m_search_modal = std::make_unique<Search<Engine::World::ObjectType::Instance>>();
		if (m_search_modal)
		{
			if (const auto [opened, instance] = m_search_modal->update(scene.mr_world); !opened || instance)
			{
				m_selected_instance = instance;
				m_search_modal.reset();
			}
		}
		ImGui::SameLine();
		ImGui::Text("%s", m_selected_instance ? m_selected_instance->name().c_str() : "<not selected>");


		const auto width = 300.0f;
		// path
		ImGui::SetNextItemWidth(-1.f);
		const bool completed = ImGui::InputTextWithHint("##object_name_input", "name",
			m_path_buffer.data(), m_path_buffer.size(), ImGuiInputTextFlags_EnterReturnsTrue);

		ImGui::SetNextItemWidth(-1.0f);
		if ((ImGui::Button("save", ImVec2(50, 0)) || completed) && m_selected_instance)
		{
			try
			{
				scene.mr_world.saver().saveOBJ(
					{m_selected_instance},
					std::filesystem::path(std::string(m_path_buffer.data())));
				m_opened = false;
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
			ImGui::TextWrapped("%s", ("Failed to save instance at specified path. Reason: " + *m_fail_message).c_str());
			ImGui::PopStyleColor();
		}
		ImGui::EndPopup();
	}

	void ModelSaveModal::update(Scene& scene)
	{
		if (!m_opened) return;

		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constinit auto* popup_id = "save model##save_model_modal_window";
		ImGui::OpenPopup(popup_id);
		if (!ImGui::BeginPopupModal(popup_id, &m_opened))
			return;

		// map selection modal
		if (ImGui::Button("Select"))
			m_search_modal = std::make_unique<Search<Engine::World::ObjectType::Group>>();
		if (m_search_modal)
		{
			if (const auto [opened, model] = m_search_modal->update(scene.mr_world); !opened || model)
			{
				m_selected_group = model;
				m_search_modal.reset();
			}
		}
		ImGui::SameLine();
		ImGui::Text("%s", m_selected_group ? m_selected_group->name().c_str() : "<not selected>");


		const auto width = 300.0f;
		// path
		ImGui::SetNextItemWidth(-1.f);
		const bool completed = ImGui::InputTextWithHint("##object_name_input", "name",
			m_path_buffer.data(), m_path_buffer.size(), ImGuiInputTextFlags_EnterReturnsTrue);

		ImGui::SetNextItemWidth(-1.0f);
		if ((ImGui::Button("save", ImVec2(50, 0)) || completed) && m_selected_group)
		{
			if (!m_selected_group->groups().empty())
				m_fail_message = "selected group as model can't have subgroups";
			else
			{
				try
				{
					scene.mr_world.saver().saveOBJ(
						m_selected_group->objects(),
						std::filesystem::path(std::string(m_path_buffer.data())));
					m_opened = false;
				}
				catch (std::exception& e)
				{
					m_fail_message = e.what();
				}
			}
		}

		if (m_fail_message)
		{
			ImGui::SetNextItemWidth(width);
			ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 64, 64, 255));
			ImGui::TextWrapped("%s", ("Failed to save model at specified path. Reason: " + *m_fail_message).c_str());
			ImGui::PopStyleColor();
		}
		ImGui::EndPopup();
	}

	void SceneSaveModal::update(Scene& scene)
	{
		const auto center = ImGui::GetMainViewport()->GetCenter();
		ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

		static constinit auto* popup_id = "save scene##save_scene_modal_window";
		if (m_opened) ImGui::OpenPopup(popup_id);
		if (ImGui::BeginPopupModal(popup_id, &m_opened, ImGuiWindowFlags_AlwaysAutoResize))
		{
			const auto width = 300.0f;
			// path
			ImGui::SetNextItemWidth(width);
			const bool completed = ImGui::InputTextWithHint("##scene_name_input", "name",
				m_path_buffer.data(), m_path_buffer.size(), ImGuiInputTextFlags_EnterReturnsTrue);
			// allow partial write
			ImGui::Checkbox("allow partial write", &m_save_options.allow_partial_write);

			ImGui::SetNextItemWidth(-1.0f);
			if (ImGui::Button("save", ImVec2(50, 0)) || completed)
			{
				try
				{
					m_save_options.path = std::filesystem::path(m_path_buffer.data());
					scene.mr_world.saver().saveScene(m_save_options);
					ImGui::CloseCurrentPopup();
					m_opened = false;
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
				ImGui::TextWrapped(("Failed to save scene at specified path. Reason: " + *m_fail_message).c_str());
				ImGui::PopStyleColor();
			}
			ImGui::EndPopup();
		}
	}
}
