#include "properties.hpp"

#include "imgui.h"

#include <numbers>
#include <array>

namespace RayZath::UI::Windows
{
	using ObjectType = Engine::World::ObjectType;

	Properties<ObjectType::Camera>::Properties(std::reference_wrapper<RZ::World> r_world)
		: PropertiesBase<ObjectType::Camera>(std::move(r_world), 120.0f)
	{}
	void Properties<ObjectType::Camera>::display()
	{
		if (!m_object) return;

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		// position
		std::array<float, 3> values3 = { m_object->GetPosition().x, m_object->GetPosition().y, m_object->GetPosition().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"position", values3.data(), 0.01f,
			-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			m_object->SetPosition(Math::vec3f(values3[0], values3[1], values3[2]));

		// rotation
		values3 = { m_object->GetRotation().x, m_object->GetRotation().y, m_object->GetRotation().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"rotation", values3.data(), 0.01f,
			-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			m_object->SetRotation(Math::vec3f(values3[0], values3[1], values3[2]));
		ImGui::NewLine();

		// resolution
		std::array<int, 2> resolution = { int(m_object->GetResolution().x), int(m_object->GetResolution().y) };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::InputInt2(
			"resolution", resolution.data(),
			ImGuiInputTextFlags_CharsDecimal))
		{
			m_object->Resize(Math::vec2ui32(
				uint32_t(std::clamp(resolution[0], 4, std::numeric_limits<int>::max())),
				uint32_t(std::clamp(resolution[1], 4, std::numeric_limits<int>::max()))));
			m_object->Focus(m_object->GetResolution() / 2);
		}
		ImGui::NewLine();

		// fov
		float fov = m_object->GetFov().value();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::SliderFloat(
			"fov", &fov,
			0.0f, std::numbers::pi_v<float> -0.01f,
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			m_object->SetFov(Math::angle_radf(fov));

		// near far
		Math::vec2f near_far = m_object->GetNearFar();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloatRange2(
			"clipping planes",
			&near_far.x, &near_far.y,
			(near_far.x + near_far.y) * 0.01f,
			0.0f, std::numeric_limits<float>::infinity()))
			m_object->SetNearFar(near_far);
		ImGui::NewLine();

		// focal distance
		float focal_distance = m_object->GetFocalDistance();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat(
			"focal distance", &focal_distance, 0.01f,
			0.0f, std::numeric_limits<float>::infinity(),
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			m_object->SetFocalDistance(focal_distance);

		// aperture
		float aperture = m_object->GetAperture();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat(
			"aperture", &aperture, 0.0001f,
			0.0f, std::numeric_limits<float>::infinity(),
			"%.4f", ImGuiSliderFlags_ClampOnInput))
			m_object->SetAperture(aperture);

		// exposure time
		float exposure_time = m_object->GetExposureTime();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat(
			"exposure time", &exposure_time, 0.001f,
			0.0f, std::numeric_limits<float>::infinity(),
			"%.3f", ImGuiSliderFlags_ClampOnInput))
			m_object->SetExposureTime(exposure_time);

		// temporal blend
		float temporal_blend = m_object->GetTemporalBlend();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::SliderFloat(
			"temporal blend", &temporal_blend,
			0.0f, 1.0f,
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			m_object->SetTemporalBlend(temporal_blend);

		// enabled
		bool render_enabled = m_object->Enabled();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::Checkbox("render enabled", &render_enabled))
			render_enabled ? m_object->EnableRender() : m_object->DisableRender();
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
		std::array<float, 3> values3 = { m_object->GetPosition().x, m_object->GetPosition().y, m_object->GetPosition().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"position", values3.data(), 0.01f,
			-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			m_object->SetPosition(Math::vec3f(values3[0], values3[1], values3[2]));

		// direction
		values3 = { m_object->GetDirection().x, m_object->GetDirection().y, m_object->GetDirection().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"direction", values3.data(), 0.01f,
			-1.0f, 1.0f,
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			m_object->SetDirection(Math::vec3f(values3[0], values3[1], values3[2]));
		ImGui::NewLine();

		// color
		std::array<float, 3> color = {
			m_object->GetColor().red / 255.0f,
			m_object->GetColor().green / 255.0f,
			m_object->GetColor().blue / 255.0f };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::ColorPicker3("color", color.data(),
			ImGuiColorEditFlags_PickerHueWheel |
			ImGuiColorEditFlags_NoLabel |
			ImGuiColorEditFlags_NoSidePreview))
			m_object->SetColor(Graphics::Color(
				uint8_t(color[0] * 255.0f),
				uint8_t(color[1] * 255.0f),
				uint8_t(color[2] * 255.0f)));
		ImGui::NewLine();

		// size
		float size = m_object->GetSize();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("size", &size, 0.01f,
			0.0f, std::numeric_limits<float>::infinity(),
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			m_object->SetSize(size);

		// emission
		float emission = m_object->GetEmission();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("emission", &emission, emission * 0.01f + 0.01f,
			0.0f, std::numeric_limits<float>::max(),
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			m_object->SetEmission(emission);

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
		std::array<float, 3> values3 = { m_object->GetDirection().x, m_object->GetDirection().y, m_object->GetDirection().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"direction", values3.data(), 0.01f,
			-1.0f, 1.0f,
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			m_object->SetDirection(Math::vec3f(values3[0], values3[1], values3[2]));
		ImGui::NewLine();

		// color
		std::array<float, 3> color = {
			m_object->GetColor().red / 255.0f,
			m_object->GetColor().green / 255.0f,
			m_object->GetColor().blue / 255.0f };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::ColorPicker3("color", color.data(),
			ImGuiColorEditFlags_PickerHueWheel |
			ImGuiColorEditFlags_NoLabel |
			ImGuiColorEditFlags_NoSidePreview))
			m_object->SetColor(Graphics::Color(
				uint8_t(color[0] * 255.0f),
				uint8_t(color[1] * 255.0f),
				uint8_t(color[2] * 255.0f)));
		ImGui::NewLine();

		// size
		float size = m_object->GetAngularSize();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("size", &size, 0.01f,
			0.0f, std::numbers::pi_v<float>,
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			m_object->SetAngularSize(size);

		// emission
		float emission = m_object->GetEmission();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("emission", &emission, emission * 0.01f + 0.01f,
			0.0f, std::numeric_limits<float>::max(),
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			m_object->SetEmission(emission);
	}

	Properties<ObjectType::Mesh>::Properties(std::reference_wrapper<RZ::World> r_world)
		: PropertiesBase<ObjectType::Mesh>(r_world)
		, m_material_properties(r_world)
	{}
	void Properties<ObjectType::Mesh>::display()
	{
		if (!m_object) return;

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		// position
		std::array<float, 3> values3 = {
			m_object->GetTransformation().GetPosition().x,
			m_object->GetTransformation().GetPosition().y,
			m_object->GetTransformation().GetPosition().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"position", values3.data(), 0.01f,
			-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			m_object->SetPosition(Math::vec3f(values3[0], values3[1], values3[2]));

		// rotation
		values3 = {
			m_object->GetTransformation().GetRotation().x,
			m_object->GetTransformation().GetRotation().y,
			m_object->GetTransformation().GetRotation().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"rotation", values3.data(), 0.01f,
			-std::numbers::pi_v<float>, std::numbers::pi_v<float>,
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			m_object->SetRotation(Math::vec3f(values3[0], values3[1], values3[2]));

		// scale
		values3 = {
			m_object->GetTransformation().GetScale().x,
			m_object->GetTransformation().GetScale().y,
			m_object->GetTransformation().GetScale().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"scale", values3.data(), 0.01f,
			std::numeric_limits<float>::epsilon(), std::numeric_limits<float>::infinity(),
			"%.3f", ImGuiSliderFlags_ClampOnInput))
			m_object->SetScale(Math::vec3f(values3[0], values3[1], values3[2]));

		ImGui::Separator();
		
		// mesh
		if (!m_selected_mesh)
			m_selected_mesh = m_object->GetStructure();

		ImGui::SetNextItemWidth(left_width);
		if (ImGui::BeginCombo(
			"mesh",
			m_selected_mesh ? m_selected_mesh->GetName().c_str() : nullptr,
			ImGuiComboFlags_HeightRegular))
		{
			auto& meshes = mr_world.get().Container<RZ::World::ObjectType::MeshStructure>();
			for (uint32_t idx = 0; idx < meshes.GetCount(); idx++)
			{
				const auto& mesh = meshes[idx];
				if (!mesh) continue;

				bool is_selected = m_selected_mesh == mesh;
				if (ImGui::Selectable(
					(mesh->GetName() + "##selectable_material" + std::to_string(idx)).c_str(), is_selected))
				{
					if (m_selected_mesh != mesh)
					{
						m_selected_mesh = mesh;
						m_object->SetMeshStructure(m_selected_mesh);
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
			for (uint32_t idx = 0; idx < m_object->GetMaterialCapacity(); idx++)
				if (m_object->GetMaterial(idx))
				{
					m_selected_material = m_object->GetMaterial(idx);
					break;
				}
		}

		ImGui::SetNextItemWidth(left_width);
		if (ImGui::BeginCombo(
			"materials",
			m_selected_material ? m_selected_material->GetName().c_str() : nullptr,
			ImGuiComboFlags_HeightRegular))
		{
			for (uint32_t idx = 0; idx < m_object->GetMaterialCapacity(); idx++)
			{
				const auto& material = m_object->GetMaterial(idx);
				if (!material) continue;

				bool is_selected = m_selected_material == material;
				if (ImGui::Selectable(
					(material->GetName() + "##selectable_material" + std::to_string(idx)).c_str(),
					is_selected))
					m_selected_material = material;

				if (is_selected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}

		if (m_selected_material)
		{
			m_material_properties.setObject(m_selected_material);
			m_material_properties.display();
		}
	}
	void Properties<ObjectType::Mesh>::reset()
	{
		m_selected_material.Release();
		m_selected_mesh.Release();
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
			m_object->transformation().GetPosition().x,
			m_object->transformation().GetPosition().y,
			m_object->transformation().GetPosition().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"position", values3.data(), 0.01f,
			-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
			"%.2f", ImGuiSliderFlags_ClampOnInput))
		{
			m_object->transformation().SetPosition(Math::vec3f(values3[0], values3[1], values3[2]));
			m_object->RequestUpdate();
		}

		// rotation
		values3 = {
			m_object->transformation().GetRotation().x,
			m_object->transformation().GetRotation().y,
			m_object->transformation().GetRotation().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"rotation", values3.data(), 0.01f,
			-std::numbers::pi_v<float>, std::numbers::pi_v<float>,
			"%.2f", ImGuiSliderFlags_ClampOnInput))
		{
			m_object->transformation().SetRotation(Math::vec3f(values3[0], values3[1], values3[2]));
			m_object->RequestUpdate();
		}

		// scale
		values3 = {
			m_object->transformation().GetScale().x,
			m_object->transformation().GetScale().y,
			m_object->transformation().GetScale().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"scale", values3.data(), 0.01f,
			std::numeric_limits<float>::epsilon(), std::numeric_limits<float>::infinity(),
			"%.3f", ImGuiSliderFlags_ClampOnInput))
		{
			m_object->transformation().SetScale(Math::vec3f(values3[0], values3[1], values3[2]));
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
		m_object.Release();
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

		// color
		std::array<float, 4> color = {
			material.GetColor().red / 255.0f,
			material.GetColor().green / 255.0f,
			material.GetColor().blue / 255.0f,
			material.GetColor().alpha / 255.0f };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::ColorPicker4("color", color.data(),
			ImGuiColorEditFlags_PickerHueWheel |
			ImGuiColorEditFlags_NoLabel |
			ImGuiColorEditFlags_AlphaBar |
			ImGuiColorEditFlags_NoSidePreview))
			material.SetColor(Graphics::Color(
				uint8_t(color[0] * 255.0f),
				uint8_t(color[1] * 255.0f),
				uint8_t(color[2] * 255.0f),
				uint8_t(color[3] * 255.0f)));
		ImGui::NewLine();

		// metalness
		float metalness = material.GetMetalness();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::SliderFloat("metalness", &metalness, 0.0f, 1.0f, "%.2f"))
			material.SetMetalness(metalness);

		// roughness
		float roughness = material.GetRoughness();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::SliderFloat(
			"roughness", &roughness,
			0.0f, 1.0f,
			"%.6f",
			ImGuiSliderFlags_Logarithmic))
			material.SetRoughness(roughness);

		// emission
		float emission = material.GetEmission();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat(
			"emission",
			&emission, emission * 0.01f + 0.01f,
			0.0f, std::numeric_limits<float>::infinity()))
			material.SetEmission(emission);

		// IOR
		float ior = material.GetIOR();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat(
			"ior",
			&ior, 0.01f,
			1.0f, std::numeric_limits<float>::infinity()))
			material.SetIOR(ior);

		// scattering
		float scattering = material.GetScattering();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat(
			"scattering",
			&scattering, 0.01f,
			0.0f, std::numeric_limits<float>::infinity()))
			material.SetScattering(scattering);
	}

	Properties<ObjectType::MeshStructure>::Properties(std::reference_wrapper<RZ::World> r_world)
		: PropertiesBase<ObjectType::MeshStructure>(std::move(r_world))
	{}
	void Properties<ObjectType::MeshStructure>::display()
	{
		if (!m_object) return;

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		ImGui::Text("vertices: ");
		ImGui::SameLine();
		ImGui::SetNextItemWidth(-FLT_MIN);
		ImGui::Text("%d", m_object->GetVertices().GetCount());

		ImGui::Text("texture coordinates: ");
		ImGui::SameLine();
		ImGui::SetNextItemWidth(-FLT_MIN);
		ImGui::Text("%d", m_object->GetTexcrds().GetCount());

		ImGui::Text("normals: ");
		ImGui::SameLine();
		ImGui::SetNextItemWidth(-FLT_MIN);
		ImGui::Text("%d", m_object->GetNormals().GetCount());

		ImGui::Text("triangles: ");
		ImGui::SameLine();
		ImGui::SetNextItemWidth(-FLT_MIN);
		ImGui::Text("%d", m_object->GetTriangles().GetCount());
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
		float translation[2] = { m_object->GetTranslation().x, m_object->GetTranslation().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("translation", translation, 0.001f))
			m_object->SetTranslation(Math::vec2f32(translation[0], translation[1]));

		// rotation
		float rotation = m_object->GetRotation().value();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("rotation", &rotation, 0.01f))
			m_object->SetRotation(Math::angle_radf(rotation));

		// scale
		float scale[2] = { m_object->GetScale().x, m_object->GetScale().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("scale", scale, 0.01f))
			m_object->SetScale(Math::vec2f32(scale[0], scale[1]));
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
		float translation[2] = { m_object->GetTranslation().x, m_object->GetTranslation().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("translation", translation, 0.001f))
			m_object->SetTranslation(Math::vec2f32(translation[0], translation[1]));

		// rotation
		float rotation = m_object->GetRotation().value();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("rotation", &rotation, 0.01f))
			m_object->SetRotation(Math::angle_radf(rotation));

		// scale
		float scale[2] = { m_object->GetScale().x, m_object->GetScale().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("scale", scale, 0.01f))
			m_object->SetScale(Math::vec2f32(scale[0], scale[1]));
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
		float translation[2] = { m_object->GetTranslation().x, m_object->GetTranslation().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("translation", translation, 0.001f))
			m_object->SetTranslation(Math::vec2f32(translation[0], translation[1]));

		// rotation
		float rotation = m_object->GetRotation().value();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("rotation", &rotation, 0.01f))
			m_object->SetRotation(Math::angle_radf(rotation));

		// scale
		float scale[2] = { m_object->GetScale().x, m_object->GetScale().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("scale", scale, 0.01f))
			m_object->SetScale(Math::vec2f32(scale[0], scale[1]));
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
		float translation[2] = { m_object->GetTranslation().x, m_object->GetTranslation().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("translation", translation, 0.001f))
			m_object->SetTranslation(Math::vec2f32(translation[0], translation[1]));

		// rotation
		float rotation = m_object->GetRotation().value();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("rotation", &rotation, 0.01f))
			m_object->SetRotation(Math::angle_radf(rotation));

		// scale
		float scale[2] = { m_object->GetScale().x, m_object->GetScale().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("scale", scale, 0.01f))
			m_object->SetScale(Math::vec2f32(scale[0], scale[1]));
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
		float translation[2] = { m_object->GetTranslation().x, m_object->GetTranslation().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("translation", translation, 0.001f))
			m_object->SetTranslation(Math::vec2f32(translation[0], translation[1]));

		// rotation
		float rotation = m_object->GetRotation().value();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("rotation", &rotation, 0.01f))
			m_object->SetRotation(Math::angle_radf(rotation));

		// scale
		float scale[2] = { m_object->GetScale().x, m_object->GetScale().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("scale", scale, 0.01f))
			m_object->SetScale(Math::vec2f32(scale[0], scale[1]));
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
