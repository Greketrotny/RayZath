#include "main_window.hpp"

#include "imgui.h"
#include "rayzath.hpp"

#include <iostream>
#include <tuple>
#include <variant>

namespace RZ = RayZath::Engine;

namespace RayZath::UI::Windows
{
	Main::Main(Scene& scene, Rendering::Module& rendering)
		: mr_scene(scene)
		, mr_rendering(rendering)
		, m_viewports(mr_scene.mr_world)
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
			auto material = mr_scene.mr_world.container<Engine::ObjectType::Material>().
				create(Engine::Material::generateMaterial<T>());
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
	template <Engine::ObjectType T> requires MapObjectType<T>
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
		m_viewports.draw(mr_rendering.m_vulkan.m_window.currentFrame().commandBuffer());

		if (auto selected_camera = m_viewports.getSelected(); selected_camera)
			m_explorer.selectObject<Engine::ObjectType::Camera>(selected_camera);
		else if (auto selected_mesh = m_viewports.getSelectedMesh(); selected_mesh)
			m_explorer.selectObject<Engine::ObjectType::Instance>(selected_mesh);
		m_explorer.update();

		m_settings.update();
		m_render_state.update();

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
			if (ImGui::MenuItem("Rendering"))
				m_render_state.open();
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu("New"))
		{
			if (ImGui::MenuItem("Camera"))
			{
				auto camera = mr_scene.mr_world.container<RZ::ObjectType::Camera>().create(
					RZ::ConStruct<RZ::Camera>("new camera"));
				m_explorer.selectObject<RZ::ObjectType::Camera>(camera);
			}
			if (ImGui::BeginMenu("Light"))
			{
				if (ImGui::MenuItem("spot"))
				{
					auto& spot_lights = mr_scene.mr_world.container<RZ::ObjectType::SpotLight>();
					auto light = spot_lights.create(RZ::ConStruct<RZ::SpotLight>("new spot light"));
					m_explorer.selectObject<ObjectType::SpotLight>(light);
				}
				if (ImGui::MenuItem("direct"))
				{
					auto lights{mr_scene.mr_world.container<RZ::ObjectType::DirectLight>()};
					auto new_light_idx = lights->add(RZ::ConStruct<RZ::DirectLight>("new direct light"));
					m_explorer.selectObject<ObjectType::DirectLight>(lights->handle(new_light_idx));
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
					mr_scene.mr_world.generateMesh<Engine::World::CommonMesh::Cube>(
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
				auto& instances = mr_scene.mr_world.container<RZ::ObjectType::Instance>();
				auto instance = instances.create(RZ::ConStruct<RZ::Instance>("new instance"));
				m_explorer.selectObject<ObjectType::Instance>(instance);
			}
			if (ImGui::MenuItem("Group"))
			{
				auto& groups = mr_scene.mr_world.container<RZ::ObjectType::Group>();
				auto group = groups.create(RZ::ConStruct<RZ::Group>("new group"));
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
			if (ImGui::MenuItem("Model"))
			{
				m_load_modals.open<LoadModal<ObjectType::Mesh>>(m_explorer);
			}
			if (ImGui::MenuItem("Scene"))
			{
				m_load_modals.open<SceneLoadModal>(m_explorer);
			}
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu("Save"))
		{
			if (ImGui::BeginMenu("Map"))
			{
				if (ImGui::MenuItem("texture"))
					m_save_modals.open<MapSaveModal<Engine::ObjectType::Texture>>();
				if (ImGui::MenuItem("normal"))
					m_save_modals.open<MapSaveModal<Engine::ObjectType::NormalMap>>();
				if (ImGui::MenuItem("metalness"))
					m_save_modals.open<MapSaveModal<Engine::ObjectType::MetalnessMap>>();
				if (ImGui::MenuItem("roughness"))
					m_save_modals.open<MapSaveModal<Engine::ObjectType::RoughnessMap>>();
				if (ImGui::MenuItem("emission"))
					m_save_modals.open<MapSaveModal<Engine::ObjectType::EmissionMap>>();
				ImGui::EndMenu();
			}
			if (ImGui::MenuItem("Material"))
			{
				m_save_modals.open<MTLSaveModal>();
			}
			if (ImGui::MenuItem("Mesh"))
			{
				m_save_modals.open<OBJSaveModal>();
			}
			if (ImGui::MenuItem("Instance"))
			{
				m_save_modals.open<InstanceSaveModal>();
			}
			if (ImGui::MenuItem("Model"))
			{
				m_save_modals.open<ModelSaveModal>();
			}
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
