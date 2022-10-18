#include "properties.hpp"

#include "explorer_base.hpp"

#include "imgui.h"
#include "rayzath.hpp"

#include <numbers>
#include <array>

namespace RayZath::UI::Windows
{
	Properties<ObjectType::Camera>::Properties(std::reference_wrapper<RZ::World> r_world)
		: PropertiesBase<ObjectType::Camera>(std::move(r_world), 120.0f)
	{}
	void Properties<ObjectType::Camera>::display()
	{
		if (!m_object) return;

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		// position
		std::array<float, 3> values3 = { m_object->position().x, m_object->position().y, m_object->position().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"position", values3.data(), 0.01f,
			-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
			"%.3f", ImGuiSliderFlags_ClampOnInput))
			m_object->position(Math::vec3f(values3[0], values3[1], values3[2]));

		// rotation
		values3 = { m_object->rotation().x, m_object->rotation().y, m_object->rotation().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"rotation", values3.data(), 0.01f,
			-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
			"%.3f", ImGuiSliderFlags_ClampOnInput))
			m_object->rotation(Math::vec3f(values3[0], values3[1], values3[2]));
		ImGui::NewLine();

		// resolution
		std::array<int, 2> resolution = { int(m_object->resolution().x), int(m_object->resolution().y) };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::InputInt2(
			"resolution", resolution.data(),
			ImGuiInputTextFlags_CharsDecimal))
		{
			m_object->resize(Math::vec2ui32(
				uint32_t(std::clamp(resolution[0], 4, std::numeric_limits<int>::max())),
				uint32_t(std::clamp(resolution[1], 4, std::numeric_limits<int>::max()))));
			m_object->focus(m_object->resolution() / 2);
		}
		ImGui::NewLine();

		// fov
		float fov = m_object->fov().value();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::SliderFloat(
			"fov", &fov,
			0.0f, std::numbers::pi_v<float> -0.01f,
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			m_object->fov(Math::angle_radf(fov));

		// near far
		Math::vec2f near_far = m_object->nearFar();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloatRange2(
			"clipping planes",
			&near_far.x, &near_far.y,
			(near_far.x + near_far.y) * 0.01f,
			0.0f, std::numeric_limits<float>::infinity()))
			m_object->nearFar(near_far);
		ImGui::NewLine();

		// focal distance
		float focal_distance = m_object->focalDistance();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat(
			"focal distance", &focal_distance, 0.01f,
			0.0f, std::numeric_limits<float>::infinity(),
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			m_object->focalDistance(focal_distance);

		// aperture
		float aperture = m_object->aperture();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat(
			"aperture", &aperture, 0.0001f,
			0.0f, std::numeric_limits<float>::infinity(),
			"%.4f", ImGuiSliderFlags_ClampOnInput))
			m_object->aperture(aperture);

		// exposure time
		float exposure_time = m_object->exposureTime();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat(
			"exposure time", &exposure_time, 0.001f,
			0.0f, std::numeric_limits<float>::infinity(),
			"%.3f", ImGuiSliderFlags_ClampOnInput))
			m_object->exposureTime(exposure_time);

		// temporal blend
		float temporal_blend = m_object->temporalBlend();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::SliderFloat(
			"temporal blend", &temporal_blend,
			0.0f, 1.0f,
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			m_object->temporalBlend(temporal_blend);

		// enabled
		bool render_enabled = m_object->enabled();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::Checkbox("render enabled", &render_enabled))
			render_enabled ? m_object->enableRender() : m_object->disableRender();
	}

	Properties<ObjectType::SpotLight>::Properties(std::reference_wrapper<RZ::World> r_world)
		: PropertiesBase<ObjectType::SpotLight>(std::move(r_world))
	{}
	void Properties<ObjectType::SpotLight>::display()
	{
		if (!m_object) return;

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		// position
		std::array<float, 3> values3 = { m_object->position().x, m_object->position().y, m_object->position().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"position", values3.data(), 0.01f,
			-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
			"%.3f", ImGuiSliderFlags_ClampOnInput))
			m_object->position(Math::vec3f(values3[0], values3[1], values3[2]));

		// direction
		values3 = { m_object->direction().x, m_object->direction().y, m_object->direction().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"direction", values3.data(), 0.01f,
			-1.0f, 1.0f,
			"%.3f", ImGuiSliderFlags_ClampOnInput))
			m_object->direction(Math::vec3f(values3[0], values3[1], values3[2]));
		ImGui::NewLine();

		// color
		std::array<float, 3> color = {
			m_object->color().red / 255.0f,
			m_object->color().green / 255.0f,
			m_object->color().blue / 255.0f };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::ColorPicker3("color", color.data(),
			ImGuiColorEditFlags_PickerHueWheel |
			ImGuiColorEditFlags_NoLabel |
			ImGuiColorEditFlags_NoSidePreview))
			m_object->color(Graphics::Color(
				uint8_t(color[0] * 255.0f),
				uint8_t(color[1] * 255.0f),
				uint8_t(color[2] * 255.0f)));
		ImGui::NewLine();

		// size
		float size = m_object->size();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("size", &size, 0.01f,
			0.0f, std::numeric_limits<float>::infinity(),
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			m_object->SetSize(size);

		// emission
		float emission = m_object->emission();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("emission", &emission, emission * 0.01f + 0.01f,
			0.0f, std::numeric_limits<float>::max(),
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			m_object->emission(emission);

		// angle
		float angle = m_object->GetBeamAngle();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("beam angle", &angle, 0.01f,
			0.0f, std::numbers::pi_v<float>,
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			m_object->SetBeamAngle(angle);
	}

	Properties<ObjectType::DirectLight>::Properties(std::reference_wrapper<RZ::World> r_world)
		: PropertiesBase<ObjectType::DirectLight>(std::move(r_world))
	{}
	void Properties<ObjectType::DirectLight>::display()
	{
		if (!m_object) return;

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		// direction
		std::array<float, 3> values3 = { m_object->direction().x, m_object->direction().y, m_object->direction().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"direction", values3.data(), 0.01f,
			-1.0f, 1.0f,
			"%.3f", ImGuiSliderFlags_ClampOnInput))
			m_object->direction(Math::vec3f(values3[0], values3[1], values3[2]));
		ImGui::NewLine();

		// color
		std::array<float, 3> color = {
			m_object->color().red / 255.0f,
			m_object->color().green / 255.0f,
			m_object->color().blue / 255.0f };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::ColorPicker3("color", color.data(),
			ImGuiColorEditFlags_PickerHueWheel |
			ImGuiColorEditFlags_NoLabel |
			ImGuiColorEditFlags_NoSidePreview))
			m_object->color(Graphics::Color(
				uint8_t(color[0] * 255.0f),
				uint8_t(color[1] * 255.0f),
				uint8_t(color[2] * 255.0f)));
		ImGui::NewLine();

		// size
		float size = m_object->angularSize();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("size", &size, 0.01f,
			0.0f, std::numbers::pi_v<float>,
			"%.3f", ImGuiSliderFlags_ClampOnInput))
			m_object->SetAngularSize(size);

		// emission
		float emission = m_object->emission();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("emission", &emission, emission * 0.01f + 0.01f,
			0.0f, std::numeric_limits<float>::max(),
			"%.3f", ImGuiSliderFlags_ClampOnInput))
			m_object->emission(emission);
	}

	Properties<ObjectType::Instance>::Properties(std::reference_wrapper<RZ::World> r_world)
		: PropertiesBase<ObjectType::Instance>(r_world)
		, m_material_properties(r_world)
	{}
	void Properties<ObjectType::Instance>::display()
	{
		if (!m_object) return;

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		// position
		std::array<float, 3> values3 = {
			m_object->transformation().position().x,
			m_object->transformation().position().y,
			m_object->transformation().position().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"position", values3.data(), 0.01f,
			-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
			"%.3f", ImGuiSliderFlags_ClampOnInput))
			m_object->position(Math::vec3f(values3[0], values3[1], values3[2]));

		// rotation
		values3 = {
			m_object->transformation().rotation().x,
			m_object->transformation().rotation().y,
			m_object->transformation().rotation().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"rotation", values3.data(), 0.01f,
			-std::numbers::pi_v<float>, std::numbers::pi_v<float>,
			"%.3f", ImGuiSliderFlags_ClampOnInput))
			m_object->rotation(Math::vec3f(values3[0], values3[1], values3[2]));

		// scale
		values3 = {
			m_object->transformation().scale().x,
			m_object->transformation().scale().y,
			m_object->transformation().scale().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"scale", values3.data(), 0.01f,
			std::numeric_limits<float>::epsilon(), std::numeric_limits<float>::infinity(),
			"%.3f", ImGuiSliderFlags_ClampOnInput))
			m_object->scale(Math::vec3f(values3[0], values3[1], values3[2]));

		ImGui::Separator();

		// mesh
		if (!m_selected_mesh)
			m_selected_mesh = m_object->mesh();

		ImGui::SetNextItemWidth(left_width);
		if (ImGui::BeginCombo(
			"mesh",
			m_selected_mesh ? m_selected_mesh->name().c_str() : nullptr,
			ImGuiComboFlags_HeightRegular))
		{
			auto& meshes = mr_world.get().container<RZ::World::ObjectType::Mesh>();
			for (uint32_t idx = 0; idx < meshes.count(); idx++)
			{
				const auto& mesh = meshes[idx];
				if (!mesh) continue;

				bool is_selected = m_selected_mesh == mesh;
				if (ImGui::Selectable(
					(mesh->name() + "##selectable_material" + std::to_string(idx)).c_str(), is_selected))
				{
					if (m_selected_mesh != mesh)
					{
						m_selected_mesh = mesh;
						m_object->mesh(m_selected_mesh);
					}
				}

				if (is_selected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}

		ImGui::Separator();

		// materials
		if (!m_selected_material)
		{
			for (uint32_t idx = 0; idx < m_object->materialCapacity(); idx++)
				if (m_object->material(idx))
				{
					m_selected_material = m_object->material(idx);
					break;
				}
		}

		ImGui::SetNextItemWidth(left_width);
		if (ImGui::BeginCombo(
			"materials",
			m_selected_material ? m_selected_material->name().c_str() : nullptr,
			ImGuiComboFlags_HeightRegular))
		{
			for (uint32_t idx = 0; idx < m_object->materialCapacity(); idx++)
			{
				const auto& material = m_object->material(idx);

				const char* material_name = material ? material->name().c_str() : "<not selected>";
				const auto action = drawSelectable(
					"#" + std::to_string(idx) + ": " +
					(std::string(material_name) + "##selectable_material" + std::to_string(idx)).c_str(),
					m_selected_material == material);

				if (action.right_clicked)
				{
					m_search_modal = std::make_unique<Search<ObjectType::Material>>();
					m_selected_material_idx = idx;
				}
				if (action.selected)
				{
					m_selected_material = material;
				}

				if (material && m_selected_material == material)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}

		if (m_search_modal)
		{
			if (const auto [opened, material] = m_search_modal->update(mr_world); !opened || material)
			{
				m_object->setMaterial(material, m_selected_material_idx);
				m_search_modal.reset();
			}
		}
		

		if (m_selected_material)
		{
			m_material_properties.setObject(m_selected_material);
			m_material_properties.display();
		}
	}
	void Properties<ObjectType::Instance>::reset()
	{
		m_selected_material.release();
		m_selected_mesh.release();
	}

	Properties<ObjectType::Group>::Properties(std::reference_wrapper<RZ::World> r_world)
		: PropertiesBase<ObjectType::Group>(std::move(r_world))
	{}
	void Properties<ObjectType::Group>::display()
	{
		if (!m_object) return;

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		// position
		std::array<float, 3> values3 = {
			m_object->transformation().position().x,
			m_object->transformation().position().y,
			m_object->transformation().position().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"position", values3.data(), 0.01f,
			-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
			"%.3f", ImGuiSliderFlags_ClampOnInput))
		{
			m_object->transformation().position(Math::vec3f(values3[0], values3[1], values3[2]));
			m_object->RequestUpdate();
		}

		// rotation
		values3 = {
			m_object->transformation().rotation().x,
			m_object->transformation().rotation().y,
			m_object->transformation().rotation().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"rotation", values3.data(), 0.01f,
			-std::numbers::pi_v<float>, std::numbers::pi_v<float>,
			"%.3f", ImGuiSliderFlags_ClampOnInput))
		{
			m_object->transformation().rotation(Math::vec3f(values3[0], values3[1], values3[2]));
			m_object->RequestUpdate();
		}

		// scale
		values3 = {
			m_object->transformation().scale().x,
			m_object->transformation().scale().y,
			m_object->transformation().scale().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"scale", values3.data(), 0.01f,
			std::numeric_limits<float>::epsilon(), std::numeric_limits<float>::infinity(),
			"%.3f", ImGuiSliderFlags_ClampOnInput))
		{
			m_object->transformation().scale(Math::vec3f(values3[0], values3[1], values3[2]));
			m_object->RequestUpdate();
		}

		ImGui::NewLine();
	}

	Properties<ObjectType::Material>::Properties(std::reference_wrapper<RZ::World> r_world)
		: PropertiesBase<ObjectType::Material>(std::move(r_world))
	{}
	void Properties<ObjectType::Material>::setObject(Engine::Material* material)
	{
		mp_material = material;
		m_object.release();
	}
	void Properties<ObjectType::Material>::display()
	{
		if (m_object)
			display(*m_object);
		else if (mp_material)
			display(*mp_material);
	}
	void Properties<ObjectType::Material>::display(RZ::Material& material)
	{
		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		ImGui::Text("parameters");
		ImGui::Separator();
		// color
		std::array<float, 4> color = {
			material.color().red / 255.0f,
			material.color().green / 255.0f,
			material.color().blue / 255.0f,
			material.color().alpha / 255.0f };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::ColorPicker4("color", color.data(),
			ImGuiColorEditFlags_PickerHueWheel |
			ImGuiColorEditFlags_AlphaBar |
			ImGuiColorEditFlags_NoSidePreview))
			material.color(Graphics::Color(
				uint8_t(color[0] * 255.0f),
				uint8_t(color[1] * 255.0f),
				uint8_t(color[2] * 255.0f),
				uint8_t(color[3] * 255.0f)));
		ImGui::NewLine();

		// metalness
		float metalness = material.metalness();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::SliderFloat("metalness", &metalness, 0.0f, 1.0f, "%.3f"))
			material.metalness(metalness);

		// roughness
		float roughness = material.roughness();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::SliderFloat(
			"roughness", &roughness,
			0.0f, 1.0f,
			"%.6f",
			ImGuiSliderFlags_Logarithmic))
			material.roughness(roughness);

		// emission
		float emission = material.emission();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat(
			"emission",
			&emission, emission * 0.01f + 0.01f,
			0.0f, std::numeric_limits<float>::infinity(), "%.3f"))
			material.emission(emission);

		// IOR
		float ior = material.ior();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat(
			"ior",
			&ior, 0.01f,
			1.0f, std::numeric_limits<float>::infinity(), "%.3f"))
			material.ior(ior);

		// scattering
		float scattering = material.scattering();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat(
			"scattering",
			&scattering, 0.01f,
			0.0f, std::numeric_limits<float>::infinity(), "%.3f"))
			material.scattering(scattering);

		// maps
		ImGui::NewLine();
		ImGui::Text("mapping");
		ImGui::Separator();

		// texture
		if (ImGui::Button("X"))
			material.texture({});
		ImGui::SameLine();
		if (ImGui::Button(
			(std::string(material.texture() ? material.texture()->name() : "<not selected>")
				+ "##texture_button").c_str(),
			ImVec2(left_width - ImGui::GetCursorPosX(), 0.0f)))
		{
			m_search_modal = Search<ObjectType::Texture>();
		}
		if (std::holds_alternative<Search<ObjectType::Texture>>(m_search_modal))
		{
			auto& search_modal = std::get<Search<ObjectType::Texture>>(m_search_modal);
			if (const auto [opened, map] = search_modal.update(mr_world); !opened || map)
			{
				material.texture(map);
				m_search_modal = std::monostate{};
			}
		}
		ImGui::SameLine();
		ImGui::Text("texture");

		// normal map
		if (ImGui::Button("X"))
			material.normalMap({});
		ImGui::SameLine();
		if (ImGui::Button(
			(std::string(material.normalMap() ? material.normalMap()->name() : "<not selected>")
				+ "##normal_button").c_str(),
			ImVec2(left_width - ImGui::GetCursorPosX(), 0.0f)))
		{
			m_search_modal = Search<ObjectType::NormalMap>();
		}
		if (std::holds_alternative<Search<ObjectType::NormalMap>>(m_search_modal))
		{
			auto& search_modal = std::get<Search<ObjectType::NormalMap>>(m_search_modal);
			if (const auto [opened, map] = search_modal.update(mr_world); !opened || map)
			{
				material.normalMap(map);
				m_search_modal = std::monostate{};
			}
		}
		ImGui::SameLine();
		ImGui::Text("normal");

		// metalness map
		if (ImGui::Button("X"))
			material.metalnessMap({});
		ImGui::SameLine();
		if (ImGui::Button(
			(std::string(material.metalnessMap() ? material.metalnessMap()->name() : "<not selected>")
				+ "##metalness_button").c_str(),
			ImVec2(left_width - ImGui::GetCursorPosX(), 0.0f)))
		{
			m_search_modal = Search<ObjectType::MetalnessMap>();
		}
		if (std::holds_alternative<Search<ObjectType::MetalnessMap>>(m_search_modal))
		{
			auto& search_modal = std::get<Search<ObjectType::MetalnessMap>>(m_search_modal);
			if (const auto [opened, map] = search_modal.update(mr_world); !opened || map)
			{
				material.metalnessMap(map);
				m_search_modal = std::monostate{};
			}
		}
		ImGui::SameLine();
		ImGui::Text("metalness");

		// roughness map
		if (ImGui::Button("X"))
			material.roughnessMap({});
		ImGui::SameLine();
		if (ImGui::Button(
			(std::string(material.roughnessMap() ? material.roughnessMap()->name() : "<not selected>")
			+ "##roughness_button").c_str(),
			ImVec2(left_width - ImGui::GetCursorPosX(), 0.0f)))
		{
			m_search_modal = Search<ObjectType::RoughnessMap>();
		}
		if (std::holds_alternative<Search<ObjectType::RoughnessMap>>(m_search_modal))
		{
			auto& search_modal = std::get<Search<ObjectType::RoughnessMap>>(m_search_modal);
			if (const auto [opened, map] = search_modal.update(mr_world); !opened || map)
			{
				material.roughnessMap(map);
				m_search_modal = std::monostate{};
			}
		}
		ImGui::SameLine();
		ImGui::Text("roughness");

		// emission map
		if (ImGui::Button("X"))
			material.emissionMap({});
		ImGui::SameLine();
		if (ImGui::Button(
			(std::string(material.emissionMap() ? material.emissionMap()->name() : "<not selected>")
				+ "##emission_button").c_str(),
			ImVec2(left_width - ImGui::GetCursorPosX(), 0.0f)))
		{
			m_search_modal = Search<ObjectType::EmissionMap>();
		}
		if (std::holds_alternative<Search<ObjectType::EmissionMap>>(m_search_modal))
		{
			auto& search_modal = std::get<Search<ObjectType::EmissionMap>>(m_search_modal);
			if (const auto [opened, map] = search_modal.update(mr_world); !opened || map)
			{
				material.emissionMap(map);
				m_search_modal = std::monostate{};
			}
		}
		ImGui::SameLine();
		ImGui::Text("emission");
	}

	Properties<ObjectType::Mesh>::Properties(std::reference_wrapper<RZ::World> r_world)
		: PropertiesBase<ObjectType::Mesh>(std::move(r_world))
	{}
	void Properties<ObjectType::Mesh>::display()
	{
		if (!m_object) return;

		ImGui::Text("vertices: ");
		ImGui::SameLine();
		ImGui::SetNextItemWidth(-FLT_MIN);
		ImGui::Text("%d", m_object->vertices().count());

		ImGui::Text("texture coordinates: ");
		ImGui::SameLine();
		ImGui::SetNextItemWidth(-FLT_MIN);
		ImGui::Text("%d", m_object->texcrds().count());

		ImGui::Text("normals: ");
		ImGui::SameLine();
		ImGui::SetNextItemWidth(-FLT_MIN);
		ImGui::Text("%d", m_object->normals().count());

		ImGui::Text("triangles: ");
		ImGui::SameLine();
		ImGui::SetNextItemWidth(-FLT_MIN);
		ImGui::Text("%d", m_object->triangles().count());
	}

	Properties<ObjectType::Texture>::Properties(std::reference_wrapper<RZ::World> r_world)
		: PropertiesBase<ObjectType::Texture>(std::move(r_world))
	{}
	void Properties<ObjectType::Texture>::display()
	{
		if (!m_object) return;

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		// translation
		float translation[2] = { m_object->translation().x, m_object->translation().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("translation", translation, 0.001f))
			m_object->translation(Math::vec2f32(translation[0], translation[1]));

		// rotation
		float rotation = m_object->rotation().value();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("rotation", &rotation, 0.01f))
			m_object->rotation(Math::angle_radf(rotation));

		// scale
		float scale[2] = { m_object->scale().x, m_object->scale().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("scale", scale, 0.01f))
			m_object->scale(Math::vec2f32(scale[0], scale[1]));
	}

	Properties<ObjectType::NormalMap>::Properties(std::reference_wrapper<RZ::World> r_world)
		: PropertiesBase<ObjectType::NormalMap>(std::move(r_world))
	{}
	void Properties<ObjectType::NormalMap>::display()
	{
		if (!m_object) return;

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		// translation
		float translation[2] = { m_object->translation().x, m_object->translation().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("translation", translation, 0.001f))
			m_object->translation(Math::vec2f32(translation[0], translation[1]));

		// rotation
		float rotation = m_object->rotation().value();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("rotation", &rotation, 0.01f))
			m_object->rotation(Math::angle_radf(rotation));

		// scale
		float scale[2] = { m_object->scale().x, m_object->scale().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("scale", scale, 0.01f))
			m_object->scale(Math::vec2f32(scale[0], scale[1]));
	}

	Properties<ObjectType::MetalnessMap>::Properties(std::reference_wrapper<RZ::World> r_world)
		: PropertiesBase<ObjectType::MetalnessMap>(std::move(r_world))
	{}
	void Properties<ObjectType::MetalnessMap>::display()
	{
		if (!m_object) return;

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		// translation
		float translation[2] = { m_object->translation().x, m_object->translation().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("translation", translation, 0.001f))
			m_object->translation(Math::vec2f32(translation[0], translation[1]));

		// rotation
		float rotation = m_object->rotation().value();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("rotation", &rotation, 0.01f))
			m_object->rotation(Math::angle_radf(rotation));

		// scale
		float scale[2] = { m_object->scale().x, m_object->scale().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("scale", scale, 0.01f))
			m_object->scale(Math::vec2f32(scale[0], scale[1]));
	}

	Properties<ObjectType::RoughnessMap>::Properties(std::reference_wrapper<RZ::World> r_world)
		: PropertiesBase<ObjectType::RoughnessMap>(std::move(r_world))
	{}
	void Properties<ObjectType::RoughnessMap>::display()
	{
		if (!m_object) return;

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		// translation
		float translation[2] = { m_object->translation().x, m_object->translation().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("translation", translation, 0.001f))
			m_object->translation(Math::vec2f32(translation[0], translation[1]));

		// rotation
		float rotation = m_object->rotation().value();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("rotation", &rotation, 0.01f))
			m_object->rotation(Math::angle_radf(rotation));

		// scale
		float scale[2] = { m_object->scale().x, m_object->scale().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("scale", scale, 0.01f))
			m_object->scale(Math::vec2f32(scale[0], scale[1]));
	}

	Properties<ObjectType::EmissionMap>::Properties(std::reference_wrapper<RZ::World> r_world)
		: PropertiesBase<ObjectType::EmissionMap>(std::move(r_world))
	{}
	void Properties<ObjectType::EmissionMap>::display()
	{
		if (!m_object) return;

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		// translation
		float translation[2] = { m_object->translation().x, m_object->translation().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("translation", translation, 0.001f))
			m_object->translation(Math::vec2f32(translation[0], translation[1]));

		// rotation
		float rotation = m_object->rotation().value();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("rotation", &rotation, 0.01f))
			m_object->rotation(Math::angle_radf(rotation));

		// scale
		float scale[2] = { m_object->scale().x, m_object->scale().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("scale", scale, 0.01f))
			m_object->scale(Math::vec2f32(scale[0], scale[1]));
	}

	void MultiProperties::displayEmpty()
	{
		ImGui::Text("no m_object to display properties of");
	}
	void MultiProperties::displayCurrentObject()
	{
		ImGui::Begin("properties");

		std::visit([](auto&& object)
			{
				if constexpr (!std::is_same_v<std::decay_t<decltype(object)>, std::monostate>)
					object.display();
			}, m_type);

		ImGui::End();
	}
}
