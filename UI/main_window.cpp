#include "main_window.hpp"

#include "imgui.h"
#include "rayzath.h"

#include <iostream>
#include <tuple>
#include <variant>

namespace RZ = RayZath::Engine;

namespace RayZath::UI::Windows
{
	Main::Main(Scene& scene)
		: mr_scene(scene)
		, m_new_modals(scene)
	{}

	using MaterialType = RayZath::Engine::Material::Common;
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
			mr_scene.mr_world.Container<Engine::World::ObjectType::Material>().Create(Engine::Material::GenerateMaterial<T>());
		}
	}

	void Main::update()
	{
		ImGui::BeginMainMenuBar();

		if (ImGui::BeginMenu("new"))
		{
			if (ImGui::MenuItem("camera"))
			{
				mr_scene.mr_world.Container<RZ::World::ObjectType::Camera>().Create(
					RZ::ConStruct<RZ::Camera>("new camera"));
			}
			if (ImGui::BeginMenu("light"))
			{
				if (ImGui::MenuItem("spot"))
				{
					auto& spot_lights = mr_scene.mr_world.Container<RZ::World::ObjectType::SpotLight>();
					spot_lights.Create(RZ::ConStruct<RZ::SpotLight>("new spot light"));
				}
				if (ImGui::MenuItem("direct"))
				{
					auto& direct_lights = mr_scene.mr_world.Container<RZ::World::ObjectType::DirectLight>();
					direct_lights.Create(RZ::ConStruct<RZ::DirectLight>("new direct light"));
				}
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("material"))
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

				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("mesh"))
			{
				if (ImGui::MenuItem("plane"))
					m_new_modals.open<NewModal<CommonMesh::Plane>>();
				if (ImGui::MenuItem("sphere"))
					m_new_modals.open<NewModal<CommonMesh::Sphere>>();
				if (ImGui::MenuItem("cone"))
					m_new_modals.open<NewModal<CommonMesh::Cone>>();
				if (ImGui::MenuItem("cylinder"))
					m_new_modals.open<NewModal<CommonMesh::Cylinder>>();
				if (ImGui::MenuItem("torus"))
					m_new_modals.open<NewModal<CommonMesh::Torus>>();
				ImGui::EndMenu();
			}
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();

		// update modals
		m_new_modals.update();
	}
}
