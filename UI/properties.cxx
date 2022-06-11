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


	void Properties::displayEmpty()
	{
		ImGui::Text("no object to display properties of");
	}
	void Properties::displayCurrentObject()
	{
		ImGui::Begin("properties");

		if (m_type.index() == 0)
			displayEmpty();
		else if (std::holds_alternative<RZ::Handle<RZ::SpotLight>>(m_type))
			SpotLightProperties::display(std::get<RZ::Handle<RZ::SpotLight>>(m_type));
		else if (std::holds_alternative<RZ::Handle<RZ::DirectLight>>(m_type))
			DirectLightProperties::display(std::get<RZ::Handle<RZ::DirectLight>>(m_type));

		ImGui::End();
	}
}



