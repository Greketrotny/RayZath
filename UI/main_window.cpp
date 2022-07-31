#include "main_window.hpp"

#include "imgui.h"
#include "rayzath.h"

#include <iostream>
#include <tuple>
#include <variant>

namespace RZ = RayZath::Engine;

namespace RayZath::UI::Windows
{
	Main::Main(Scene& scene, Rendering::Module& rendering)
		: mr_scene(scene)
		, mr_rendering(rendering)
		, m_explorer(scene, m_viewports)
		, m_new_modals(scene)
		, m_load_modals(scene)
		, m_save_modals(scene)
	{}

	using MaterialType = RayZath::Engine::Material::Common;
	using static_dictionary = RayZath::Utils::static_dictionary;
	using namespace std::string_view_literals;
	template <MaterialType T>
	constexpr auto material_name = static_dictionary::vv_translate<T>::template with<
		static_dictionary::vv_translation<MaterialType::Gold, 0>,
		static_dictionary::vv_translation<MaterialType::Silver, 1>,
		static_dictionary::vv_translation<MaterialType::Copper, 2>,
		static_dictionary::vv_translation<MaterialType::Glass, 3>,
		static_dictionary::vv_translation<MaterialType::Water, 4>,
		static_dictionary::vv_translation<MaterialType::Mirror, 5>,
		static_dictionary::vv_translation<MaterialType::RoughWood, 6>,
		static_dictionary::vv_translation<MaterialType::PolishedWood, 7>,
		static_dictionary::vv_translation<MaterialType::Paper, 8>,
		static_dictionary::vv_translation<MaterialType::Rubber, 9>,
		static_dictionary::vv_translation<MaterialType::RoughPlastic, 10>,
		static_dictionary::vv_translation<MaterialType::PolishedPlastic, 11>,
		static_dictionary::vv_translation<MaterialType::Porcelain, 12>>::template value;
	constexpr std::array material_names = {
		"gold"sv,
		"silver"sv,
		"copper"sv,
		"glass"sv,
		"water"sv,
		"mirror"sv,
		"rough wood"sv,
		"polished wood"sv,
		"paper"sv,
		"rubber"sv,
		"rough plastic"sv,
		"polished plastic"sv,
		"porcelain"sv
	};
	template <MaterialType T>
	constexpr auto materialName()
	{
		constexpr auto idx = material_name<T>;
		static_assert(idx < material_names.size());
		return material_names[idx];
	}
	template <Engine::Material::Common T>
	void Main::materialItem()
	{
		static constexpr auto name = materialName<T>();
		if (ImGui::MenuItem(name.data()))
		{
			auto material = mr_scene.mr_world.Container<Engine::World::ObjectType::Material>().
				Create(Engine::Material::GenerateMaterial<T>());
			m_explorer.selectObject<ObjectType::Material>(material);
		}
	}

	template <ObjectType T>
	constexpr auto map_name_idx = static_dictionary::vv_translate<T>::template with<
		static_dictionary::vv_translation<ObjectType::Texture, 0>,
		static_dictionary::vv_translation<ObjectType::NormalMap, 1>,
		static_dictionary::vv_translation<ObjectType::MetalnessMap, 2>,
		static_dictionary::vv_translation<ObjectType::RoughnessMap, 3>,
		static_dictionary::vv_translation<ObjectType::EmissionMap, 4>>::template value;
	constexpr std::array map_names = {
		"texture"sv,
		"normal"sv,
		"metalness"sv,
		"roughness"sv,
		"emission"sv
	};
	template <ObjectType T>
	constexpr auto mapName()
	{
		constexpr auto idx = map_name_idx<T>;
		static_assert(idx < map_names.size());
		return map_names[idx];
	}
	template <Engine::World::ObjectType T> requires MapObjectType<T>
	void Main::mapItem()
	{
		static constexpr auto name = mapName<T>();
		if (ImGui::MenuItem(name.data()))
		{
			m_load_modals.open<LoadModal<T>>(m_explorer);
		}
	}

	void Main::update()
	{
		m_viewports.destroyInvalidViewports();

		m_viewports.update(mr_rendering.m_vulkan.m_window.currentFrame().commandBuffer());
		m_viewports.draw();

		if (auto selected_camera = m_viewports.getSelected(); selected_camera)
			m_explorer.selectObject<Engine::World::ObjectType::Camera>(selected_camera);

		m_explorer.update();

		m_settings.update();

		updateMenuBar();
	}

	void Main::updateMenuBar()
	{
		ImGui::BeginMainMenuBar();

		if (ImGui::BeginMenu("Window"))
		{
			if (ImGui::MenuItem("Settings"))
				m_settings.open();
			if (ImGui::MenuItem("Explorer"))
				m_explorer.open();
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu("New"))
		{
			if (ImGui::MenuItem("Camera"))
			{
				auto camera = mr_scene.mr_world.Container<RZ::World::ObjectType::Camera>().Create(
					RZ::ConStruct<RZ::Camera>("new camera"));
				m_explorer.selectObject<RZ::World::ObjectType::Camera>(camera);
			}
			if (ImGui::BeginMenu("Light"))
			{
				if (ImGui::MenuItem("spot"))
				{
					auto& spot_lights = mr_scene.mr_world.Container<RZ::World::ObjectType::SpotLight>();
					auto light = spot_lights.Create(RZ::ConStruct<RZ::SpotLight>("new spot light"));
					m_explorer.selectObject<ObjectType::SpotLight>(light);
				}
				if (ImGui::MenuItem("direct"))
				{
					auto& direct_lights = mr_scene.mr_world.Container<RZ::World::ObjectType::DirectLight>();
					auto light = direct_lights.Create(RZ::ConStruct<RZ::DirectLight>("new direct light"));
					m_explorer.selectObject<ObjectType::DirectLight>(light);
				}
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("Material"))
			{
				materialItem<MaterialType::Gold>();
				materialItem<MaterialType::Silver>();
				materialItem<MaterialType::Copper>();
				materialItem<MaterialType::Glass>();
				materialItem<MaterialType::Water>();
				materialItem<MaterialType::Mirror>();
				materialItem<MaterialType::RoughWood>();
				materialItem<MaterialType::PolishedWood>();
				materialItem<MaterialType::Paper>();
				materialItem<MaterialType::Rubber>();
				materialItem<MaterialType::RoughPlastic>();
				materialItem<MaterialType::PolishedPlastic>();
				materialItem<MaterialType::Porcelain>();

				if (ImGui::MenuItem("custom"))
					m_new_modals.open<NewMaterialModal>(m_explorer);

				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("Mesh"))
			{
				if (ImGui::MenuItem("cube"))
					mr_scene.mr_world.GenerateMesh<Engine::World::CommonMesh::Cube>(
						Engine::World::CommonMeshParameters<Engine::World::CommonMesh::Cube>{});
				if (ImGui::MenuItem("plane"))
					m_new_modals.open<NewMeshModal<Engine::World::CommonMesh::Plane>>(m_explorer);
				if (ImGui::MenuItem("sphere"))
					m_new_modals.open<NewMeshModal<Engine::World::CommonMesh::Sphere>>(m_explorer);
				if (ImGui::MenuItem("cone"))
					m_new_modals.open<NewMeshModal<Engine::World::CommonMesh::Cone>>(m_explorer);
				if (ImGui::MenuItem("cylinder"))
					m_new_modals.open<NewMeshModal<Engine::World::CommonMesh::Cylinder>>(m_explorer);
				if (ImGui::MenuItem("torus"))
					m_new_modals.open<NewMeshModal<Engine::World::CommonMesh::Torus>>(m_explorer);
				ImGui::EndMenu();
			}
			if (ImGui::MenuItem("Instance"))
			{
				auto& instances = mr_scene.mr_world.Container<RZ::World::ObjectType::Mesh>();
				auto instance = instances.Create(RZ::ConStruct<RZ::Mesh>("new instance"));
				m_explorer.selectObject<ObjectType::Mesh>(instance);
			}
			if (ImGui::MenuItem("Group"))
			{
				auto& groups = mr_scene.mr_world.Container<RZ::World::ObjectType::Group>();
				auto group = groups.Create(RZ::ConStruct<RZ::Group>("new group"));
				m_explorer.selectObject<ObjectType::Group>(group);
			}
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu("Load"))
		{
			if (ImGui::BeginMenu("Map"))
			{
				mapItem<ObjectType::Texture>();
				mapItem<ObjectType::NormalMap>();
				mapItem<ObjectType::MetalnessMap>();
				mapItem<ObjectType::RoughnessMap>();
				mapItem<ObjectType::EmissionMap>();
				ImGui::EndMenu();
			}
			if (ImGui::MenuItem("Material"))
			{
				m_load_modals.open<LoadModal<ObjectType::Material>>(m_explorer);
			}
			if (ImGui::MenuItem("Mesh"))
			{
				m_load_modals.open<LoadModal<ObjectType::MeshStructure>>(m_explorer);
			}
			if (ImGui::MenuItem("Scene"))
			{
				m_load_modals.open<SceneLoadModal>(m_explorer);
			}
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu("Save"))
		{
			if (ImGui::MenuItem("Scene"))
			{
				m_save_modals.open<SceneSaveModal>();
			}
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();

		// update modals
		m_new_modals.update();
		m_load_modals.update();
		m_save_modals.update();
	}
}
