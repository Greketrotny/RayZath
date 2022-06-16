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
		std::array<int, 2> resolution = { camera->GetResolution().x, camera->GetResolution().y };
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
			0.0f, std::numbers::pi_v<float> - 0.01f,
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
		const float content_width = ImGui::GetContentRegionAvail().x;
		const float left_width = content_width - m_label_width;

		// position
		std::array<float, 3> values3 = { 
			object->GetTransformation().GetPosition().x, 
			object->GetTransformation().GetPosition().y,
			object->GetTransformation().GetPosition().z};
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
		if (ImGui::SliderFloat3(
			"rotation", values3.data(),
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
	}

	void Properties::displayEmpty()
	{
		ImGui::Text("no object to display properties of");
	}
	void Properties::displayCurrentObject()
	{
		ImGui::Begin("properties");

		if (m_type.index() == 0)
			displayEmpty();
		else if (std::holds_alternative<RZ::Handle<RZ::Camera>>(m_type))
			CameraProperties{}.display(std::get<RZ::Handle<RZ::Camera>>(m_type));
		else if (std::holds_alternative<RZ::Handle<RZ::SpotLight>>(m_type))
			SpotLightProperties{}.display(std::get<RZ::Handle<RZ::SpotLight>>(m_type));
		else if (std::holds_alternative<RZ::Handle<RZ::DirectLight>>(m_type))
			DirectLightProperties{}.display(std::get<RZ::Handle<RZ::DirectLight>>(m_type));
		else if (std::holds_alternative<RZ::Handle<RZ::Mesh>>(m_type))
			MeshProperties{}.display(std::get<RZ::Handle<RZ::Mesh>>(m_type));

		ImGui::End();
	}
}



