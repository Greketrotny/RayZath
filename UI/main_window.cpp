#include "main_window.hpp"

#include "imgui.h"
#include "rayzath.h"

#include "explorer.hpp"

#include <iostream>
#include <tuple>
#include <variant>

namespace RZ = RayZath::Engine;

namespace RayZath::UI::Windows
{
	Main::Main(Scene& scene)
		: mr_scene(scene)
		, m_new_modals(scene)
		, m_load_modals(scene)
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
	void Main::materialItem(SceneExplorer& explorer)
	{
		static constexpr auto name = materialName<T>();
		if (ImGui::MenuItem(name.data()))
		{
			auto material = mr_scene.mr_world.Container<Engine::World::ObjectType::Material>().
				Create(Engine::Material::GenerateMaterial<T>());
			explorer.selectObject<ObjectType::Material>(material);
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
	void Main::mapItem(SceneExplorer& explorer)
	{
		static constexpr auto name = mapName<T>();
		if (ImGui::MenuItem(name.data()))
		{
			m_load_modals.open<LoadModal<T>>(explorer);
		}
	}

	void Main::update(SceneExplorer& explorer)
	{
		ImGui::BeginMainMenuBar();

		if (ImGui::BeginMenu("new"))
		{
			if (ImGui::MenuItem("camera"))
			{
				auto camera = mr_scene.mr_world.Container<RZ::World::ObjectType::Camera>().Create(
					RZ::ConStruct<RZ::Camera>("new camera"));
				explorer.selectObject<RZ::World::ObjectType::Camera>(camera);
			}
			if (ImGui::BeginMenu("light"))
			{
				if (ImGui::MenuItem("spot"))
				{
					auto& spot_lights = mr_scene.mr_world.Container<RZ::World::ObjectType::SpotLight>();
					auto light = spot_lights.Create(RZ::ConStruct<RZ::SpotLight>("new spot light"));
					explorer.selectObject<ObjectType::SpotLight>(light);
				}
				if (ImGui::MenuItem("direct"))
				{
					auto& direct_lights = mr_scene.mr_world.Container<RZ::World::ObjectType::DirectLight>();
					auto light = direct_lights.Create(RZ::ConStruct<RZ::DirectLight>("new direct light"));
					explorer.selectObject<ObjectType::DirectLight>(light);
				}
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("material"))
			{
				materialItem<MaterialType::Gold>(explorer);
				materialItem<MaterialType::Silver>(explorer);
				materialItem<MaterialType::Copper>(explorer);
				materialItem<MaterialType::Glass>(explorer);
				materialItem<MaterialType::Water>(explorer);
				materialItem<MaterialType::Mirror>(explorer);
				materialItem<MaterialType::RoughWood>(explorer);
				materialItem<MaterialType::PolishedWood>(explorer);
				materialItem<MaterialType::Paper>(explorer);
				materialItem<MaterialType::Rubber>(explorer);
				materialItem<MaterialType::RoughPlastic>(explorer);
				materialItem<MaterialType::PolishedPlastic>(explorer);
				materialItem<MaterialType::Porcelain>(explorer);

				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("mesh"))
			{
				if (ImGui::MenuItem("plane"))
					m_new_modals.open<NewModal<CommonMesh::Plane>>(explorer);
				if (ImGui::MenuItem("sphere"))
					m_new_modals.open<NewModal<CommonMesh::Sphere>>(explorer);
				if (ImGui::MenuItem("cone"))
					m_new_modals.open<NewModal<CommonMesh::Cone>>(explorer);
				if (ImGui::MenuItem("cylinder"))
					m_new_modals.open<NewModal<CommonMesh::Cylinder>>(explorer);
				if (ImGui::MenuItem("torus"))
					m_new_modals.open<NewModal<CommonMesh::Torus>>(explorer);
				ImGui::EndMenu();
			}
			if (ImGui::MenuItem("instance"))
			{
				auto& instances = mr_scene.mr_world.Container<RZ::World::ObjectType::Mesh>();
				auto instance = instances.Create(RZ::ConStruct<RZ::Mesh>("new instance"));
				explorer.selectObject<ObjectType::Mesh>(instance);
			}
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu("load"))
		{
			if (ImGui::BeginMenu("map"))
			{
				mapItem<ObjectType::Texture>(explorer);
				mapItem<ObjectType::NormalMap>(explorer);
				mapItem<ObjectType::MetalnessMap>(explorer);
				mapItem<ObjectType::RoughnessMap>(explorer);
				mapItem<ObjectType::EmissionMap>(explorer);
				ImGui::EndMenu();
			}
			if (ImGui::MenuItem("material"))
			{
				m_load_modals.open<LoadModal<ObjectType::Material>>(explorer);
			}
			if (ImGui::MenuItem("mesh"))
			{
				m_load_modals.open<LoadModal<ObjectType::MeshStructure>>(explorer);
			}
			if (ImGui::MenuItem("scene"))
			{
				m_load_modals.open<SceneLoadModal>(explorer);
			}
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();

		// update modals
		m_new_modals.update();
		m_load_modals.update();
	}
}
