#include "settings.hpp"

#include "rayzath.hpp"

#include "imgui.h"

namespace RayZath::UI::Windows
{
	void Settings::open()
	{
		m_opened = true;
	}
	bool Settings::isOpened() const
	{
		return m_opened;
	}

	void Settings::update()
	{
		if (!m_opened) return;
		if (!ImGui::Begin("settings", &m_opened))
		{
			ImGui::End();
			return;
		}

		const float content_width = ImGui::GetContentRegionAvail().x;
		const float label_width = 100.0f;
		const float left_width = content_width - label_width;

		ImGui::Text("direct light sampling");
		ImGui::Separator();

		// spot light		
		int spot_light_n = RayZath::Engine::Engine::GetInstance().GetRenderConfig().GetLightSampling().GetSpotLight();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragInt("spot light", &spot_light_n, 0.1f, 0, 255, "%d"))
			RayZath::Engine::Engine::GetInstance().GetRenderConfig().GetLightSampling().SetSpotLight(uint8_t(spot_light_n));
		// direct light		
		int direct_light_n = RayZath::Engine::Engine::GetInstance().GetRenderConfig().GetLightSampling().GetDirectLight();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragInt("direct light", &direct_light_n, 0.1f, 0, 255, "%d"))
			RayZath::Engine::Engine::GetInstance().GetRenderConfig().GetLightSampling().SetDirectLight(uint8_t(direct_light_n));


		ImGui::NewLine();
		ImGui::Text("tracing");
		ImGui::Separator();
		int max_depth = RayZath::Engine::Engine::GetInstance().GetRenderConfig().GetTracing().GetMaxDepth();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragInt("max depth", &max_depth, 0.1f, 1, 255, "%d"))
			RayZath::Engine::Engine::GetInstance().GetRenderConfig().GetTracing().SetMaxDepth(uint8_t(max_depth));
		int rpp = RayZath::Engine::Engine::GetInstance().GetRenderConfig().GetTracing().GetRPP();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragInt("rays per pixel", &rpp, 0.1f, 1, 255, "%d"))
			RayZath::Engine::Engine::GetInstance().GetRenderConfig().GetTracing().SetRPP(uint8_t(rpp));

		ImGui::End();
	}
}