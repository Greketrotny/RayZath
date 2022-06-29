module;

#include "imgui.h"

#include "rayzath.h"

#include <variant>
#include <numbers>

module rz.ui.windows.properties;


namespace RZ = RayZath::Engine;

namespace RayZath::UI::Windows
{
	PropertiesBase::PropertiesBase(float label_width)
		: m_label_width(label_width)
	{}


	void SpotLightProperties::display(const RZ::Handle<RZ::SpotLight>& light)
	{
		if (!light) return;

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		// position
		std::array<float, 3> values3 = { light->GetPosition().x, light->GetPosition().y, light->GetPosition().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"position", values3.data(), 0.01f,
			-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			light->SetPosition(Math::vec3f(values3[0], values3[1], values3[2]));

		// direction
		values3 = { light->GetDirection().x, light->GetDirection().y, light->GetDirection().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"direction", values3.data(), 0.01f,
			-1.0f, 1.0f,
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			light->SetDirection(Math::vec3f(values3[0], values3[1], values3[2]));
		ImGui::NewLine();

		// color
		std::array<float, 3> color = {
			light->GetColor().red / 255.0f,
			light->GetColor().green / 255.0f,
			light->GetColor().blue / 255.0f };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::ColorPicker3("color", color.data(),
			ImGuiColorEditFlags_PickerHueWheel |
			ImGuiColorEditFlags_NoLabel |
			ImGuiColorEditFlags_NoSidePreview))
			light->SetColor(Graphics::Color(
				uint8_t(color[0] * 255.0f),
				uint8_t(color[1] * 255.0f),
				uint8_t(color[2] * 255.0f)));
		ImGui::NewLine();

		// size
		float size = light->GetSize();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("size", &size, 0.01f,
			0.0f, std::numeric_limits<float>::infinity(),
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			light->SetSize(size);

		// emission
		float emission = light->GetEmission();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("emission", &emission, emission * 0.01f + 0.01f,
			0.0f, std::numeric_limits<float>::max(),
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			light->SetEmission(emission);

		// angle
		float angle = light->GetBeamAngle();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("beam angle", &angle, 0.01f,
			0.0f, std::numbers::pi_v<float>,
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			light->SetBeamAngle(angle);
	}
	void DirectLightProperties::display(const RZ::Handle<RZ::DirectLight>& light)
	{
		if (!light) return;

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		// direction
		std::array<float, 3> values3 = { light->GetDirection().x, light->GetDirection().y, light->GetDirection().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"direction", values3.data(), 0.01f,
			-1.0f, 1.0f,
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			light->SetDirection(Math::vec3f(values3[0], values3[1], values3[2]));
		ImGui::NewLine();

		// color
		std::array<float, 3> color = {
			light->GetColor().red / 255.0f,
			light->GetColor().green / 255.0f,
			light->GetColor().blue / 255.0f };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::ColorPicker3("color", color.data(),
			ImGuiColorEditFlags_PickerHueWheel |
			ImGuiColorEditFlags_NoLabel |
			ImGuiColorEditFlags_NoSidePreview))
			light->SetColor(Graphics::Color(
				uint8_t(color[0] * 255.0f),
				uint8_t(color[1] * 255.0f),
				uint8_t(color[2] * 255.0f)));
		ImGui::NewLine();

		// size
		float size = light->GetAngularSize();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("size", &size, 0.01f,
			0.0f, std::numbers::pi_v<float>,
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			light->SetAngularSize(size);

		// emission
		float emission = light->GetEmission();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("emission", &emission, emission * 0.01f + 0.01f,
			0.0f, std::numeric_limits<float>::max(),
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			light->SetEmission(emission);
	}
	void CameraProperties::display(const RZ::Handle<RZ::Camera>& camera)
	{
		if (!camera) return;

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		// position
		std::array<float, 3> values3 = { camera->GetPosition().x, camera->GetPosition().y, camera->GetPosition().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"position", values3.data(), 0.01f,
			-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			camera->SetPosition(Math::vec3f(values3[0], values3[1], values3[2]));

		// rotation
		values3 = { camera->GetRotation().x, camera->GetRotation().y, camera->GetRotation().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"rotation", values3.data(), 0.01f,
			-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			camera->SetRotation(Math::vec3f(values3[0], values3[1], values3[2]));
		ImGui::NewLine();

		// resolution
		std::array<int, 2> resolution = { int(camera->GetResolution().x), int(camera->GetResolution().y) };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::InputInt2(
			"resolution", resolution.data(),
			ImGuiInputTextFlags_CharsDecimal))
		{
			camera->Resize(Math::vec2ui32(
				uint32_t(std::clamp(resolution[0], 4, std::numeric_limits<int>::max())),
				uint32_t(std::clamp(resolution[1], 4, std::numeric_limits<int>::max()))));
			camera->Focus(camera->GetResolution() / 2);
		}
		ImGui::NewLine();

		// fov
		float fov = camera->GetFov().value();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::SliderFloat(
			"fov", &fov,
			0.0f, std::numbers::pi_v<float> -0.01f,
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			camera->SetFov(Math::angle_radf(fov));

		// near far
		Math::vec2f near_far = camera->GetNearFar();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloatRange2(
			"clipping planes",
			&near_far.x, &near_far.y,
			(near_far.x + near_far.y) * 0.01f,
			0.0f, std::numeric_limits<float>::infinity()))
			camera->SetNearFar(near_far);
		ImGui::NewLine();

		// focal distance
		float focal_distance = camera->GetFocalDistance();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat(
			"focal distance", &focal_distance, 0.01f,
			0.0f, std::numeric_limits<float>::infinity(),
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			camera->SetFocalDistance(focal_distance);

		// aperture
		float aperture = camera->GetAperture();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat(
			"aperture", &aperture, 0.0001f,
			0.0f, std::numeric_limits<float>::infinity(),
			"%.4f", ImGuiSliderFlags_ClampOnInput))
			camera->SetAperture(aperture);

		// exposure time
		float exposure_time = camera->GetExposureTime();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat(
			"exposure time", &exposure_time, 0.001f,
			0.0f, std::numeric_limits<float>::infinity(),
			"%.3f", ImGuiSliderFlags_ClampOnInput))
			camera->SetExposureTime(exposure_time);

		// temporal blend
		float temporal_blend = camera->GetTemporalBlend();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::SliderFloat(
			"temporal blend", &temporal_blend,
			0.0f, 1.0f,
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			camera->SetTemporalBlend(temporal_blend);

		// enabled
		bool render_enabled = camera->Enabled();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::Checkbox("render enabled", &render_enabled))
			render_enabled ? camera->EnableRender() : camera->DisableRender();
	}
	void MeshProperties::display(const RZ::Handle<RZ::Mesh>& object)
	{
		if (!object) return;

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		// position
		std::array<float, 3> values3 = {
			object->GetTransformation().GetPosition().x,
			object->GetTransformation().GetPosition().y,
			object->GetTransformation().GetPosition().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"position", values3.data(), 0.01f,
			-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			object->SetPosition(Math::vec3f(values3[0], values3[1], values3[2]));

		// rotation
		values3 = {
			object->GetTransformation().GetRotation().x,
			object->GetTransformation().GetRotation().y,
			object->GetTransformation().GetRotation().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"rotation", values3.data(), 0.01f,
			-std::numbers::pi_v<float>, std::numbers::pi_v<float>,
			"%.2f", ImGuiSliderFlags_ClampOnInput))
			object->SetRotation(Math::vec3f(values3[0], values3[1], values3[2]));

		// scale
		values3 = {
			object->GetTransformation().GetScale().x,
			object->GetTransformation().GetScale().y,
			object->GetTransformation().GetScale().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"scale", values3.data(), 0.01f,
			std::numeric_limits<float>::epsilon(), std::numeric_limits<float>::infinity(),
			"%.3f", ImGuiSliderFlags_ClampOnInput))
			object->SetScale(Math::vec3f(values3[0], values3[1], values3[2]));

		ImGui::NewLine();

		if (!m_selected_material)
		{
			for (uint32_t idx = 0; idx < object->GetMaterialCapacity(); idx++)
				if (object->GetMaterial(idx))
				{
					m_selected_material = object->GetMaterial(idx);
					break;
				}
		}

		// materials
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::BeginCombo(
			"materials",
			m_selected_material ? m_selected_material->GetName().c_str() : nullptr,
			ImGuiComboFlags_HeightRegular))
		{
			for (uint32_t idx = 0; idx < object->GetMaterialCapacity(); idx++)
			{
				const auto& material = object->GetMaterial(idx);
				if (!material) continue;

				bool is_selected = m_selected_material == material;
				if (ImGui::Selectable(material->GetName().c_str(), is_selected))
					m_selected_material = material;

				if (is_selected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}

		if (m_selected_material)
			MaterialProperties{}.display(m_selected_material);
	}
	void GroupProperties::display(const RZ::Handle<RZ::Group>& group)
	{
		if (!group) return;

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		// position
		std::array<float, 3> values3 = {
			group->transformation().GetPosition().x,
			group->transformation().GetPosition().y,
			group->transformation().GetPosition().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"position", values3.data(), 0.01f,
			-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
			"%.2f", ImGuiSliderFlags_ClampOnInput))
		{
			group->transformation().SetPosition(Math::vec3f(values3[0], values3[1], values3[2]));
			group->RequestUpdate();
		}

		// rotation
		values3 = {
			group->transformation().GetRotation().x,
			group->transformation().GetRotation().y,
			group->transformation().GetRotation().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"rotation", values3.data(), 0.01f,
			-std::numbers::pi_v<float>, std::numbers::pi_v<float>,
			"%.2f", ImGuiSliderFlags_ClampOnInput))
		{
			group->transformation().SetRotation(Math::vec3f(values3[0], values3[1], values3[2]));
			group->RequestUpdate();
		}

		// scale
		values3 = {
			group->transformation().GetScale().x,
			group->transformation().GetScale().y,
			group->transformation().GetScale().z };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat3(
			"scale", values3.data(), 0.01f,
			std::numeric_limits<float>::epsilon(), std::numeric_limits<float>::infinity(),
			"%.3f", ImGuiSliderFlags_ClampOnInput))
		{
			group->transformation().SetScale(Math::vec3f(values3[0], values3[1], values3[2]));
			group->RequestUpdate();
		}

		ImGui::NewLine();
	}

	void MaterialProperties::display(const RZ::Handle<RZ::Material>& material)
	{
		if (!material) return;
		display(*material);
	}
	void MaterialProperties::display(RZ::Material& material)
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

	void TextureProperties::display(const RZ::Handle<RZ::Texture>& texture)
	{
		if (!texture) return;

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		// translation
		float translation[2] = { texture->GetTranslation().x, texture->GetTranslation().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("translation", translation, 0.001f))
			texture->SetTranslation(Math::vec2f32(translation[0], translation[1]));

		// rotation
		float rotation = texture->GetRotation().value();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("rotation", &rotation, 0.01f))
			texture->SetRotation(Math::angle_radf(rotation));

		// scale
		float scale[2] = { texture->GetScale().x, texture->GetScale().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("scale", scale, 0.01f))
			texture->SetScale(Math::vec2f32(scale[0], scale[1]));
	}
	void NormalMapProperties::display(const RZ::Handle<RZ::NormalMap>& map)
	{
		if (!map) return;

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		// translation
		float translation[2] = { map->GetTranslation().x, map->GetTranslation().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("translation", translation, 0.001f))
			map->SetTranslation(Math::vec2f32(translation[0], translation[1]));

		// rotation
		float rotation = map->GetRotation().value();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("rotation", &rotation, 0.01f))
			map->SetRotation(Math::angle_radf(rotation));

		// scale
		float scale[2] = { map->GetScale().x, map->GetScale().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("scale", scale, 0.01f))
			map->SetScale(Math::vec2f32(scale[0], scale[1]));
	}
	void MetalnessMapProperties::display(const RZ::Handle<RZ::MetalnessMap>& map)
	{
		if (!map) return;

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		// translation
		float translation[2] = { map->GetTranslation().x, map->GetTranslation().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("translation", translation, 0.001f))
			map->SetTranslation(Math::vec2f32(translation[0], translation[1]));

		// rotation
		float rotation = map->GetRotation().value();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("rotation", &rotation, 0.01f))
			map->SetRotation(Math::angle_radf(rotation));

		// scale
		float scale[2] = { map->GetScale().x, map->GetScale().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("scale", scale, 0.01f))
			map->SetScale(Math::vec2f32(scale[0], scale[1]));
	}
	void RoughnessMapProperties::display(const RZ::Handle<RZ::RoughnessMap>& map)
	{
		if (!map) return;

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		// translation
		float translation[2] = { map->GetTranslation().x, map->GetTranslation().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("translation", translation, 0.001f))
			map->SetTranslation(Math::vec2f32(translation[0], translation[1]));

		// rotation
		float rotation = map->GetRotation().value();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("rotation", &rotation, 0.01f))
			map->SetRotation(Math::angle_radf(rotation));

		// scale
		float scale[2] = { map->GetScale().x, map->GetScale().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("scale", scale, 0.01f))
			map->SetScale(Math::vec2f32(scale[0], scale[1]));
	}
	void EmissionMapProperties::display(const RZ::Handle<RZ::EmissionMap>& map)
	{
		if (!map) return;

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		// translation
		float translation[2] = { map->GetTranslation().x, map->GetTranslation().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("translation", translation, 0.001f))
			map->SetTranslation(Math::vec2f32(translation[0], translation[1]));

		// rotation
		float rotation = map->GetRotation().value();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat("rotation", &rotation, 0.01f))
			map->SetRotation(Math::angle_radf(rotation));

		// scale
		float scale[2] = { map->GetScale().x, map->GetScale().y };
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragFloat2("scale", scale, 0.01f))
			map->SetScale(Math::vec2f32(scale[0], scale[1]));
	}

	void Properties::displayEmpty()
	{
		ImGui::Text("no object to display properties of");
	}
	void Properties::displayCurrentObject()
	{
		ImGui::Begin("properties");

		if (m_material)
			MaterialProperties::display(*m_material);

		else if (m_type.index() == 0)
			displayEmpty();
		else if (m_type.index() == 1)
			CameraProperties::display(std::get<1>(m_type));
		else if (m_type.index() == 2)
			SpotLightProperties::display(std::get<2>(m_type));
		else if (m_type.index() == 3)
			DirectLightProperties::display(std::get<3>(m_type));
		else if (m_type.index() == 4)
			MeshProperties::display(std::get<4>(m_type));
		else if (m_type.index() == 5)
			GroupProperties::display(std::get<5>(m_type));
		else if (m_type.index() == 6)
			MaterialProperties::display(std::get<6>(m_type));

		else if (m_type.index() == 7)
			TextureProperties::display(std::get<7>(m_type));
		else if (m_type.index() == 8)
			NormalMapProperties::display(std::get<8>(m_type));
		else if (m_type.index() == 9)
			MetalnessMapProperties::display(std::get<9>(m_type));
		else if (m_type.index() == 10)
			RoughnessMapProperties::display(std::get<10>(m_type));
		else if (m_type.index() == 11)
			EmissionMapProperties::display(std::get<11>(m_type));

		ImGui::End();
	}
}



