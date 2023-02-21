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

		std::map<RayZath::Engine::Engine::RenderEngine, const char*> engines = {
			{RayZath::Engine::Engine::RenderEngine::CPU, "CPU"},
			{RayZath::Engine::Engine::RenderEngine::CUDAGPU, "CUDA"}
		};
		auto current = RayZath::Engine::Engine::instance().renderEngine();
		if (ImGui::BeginCombo("rendering engine###engine_type", engines[current]))
		{
			for (const auto& [engine, name] : engines)
			{
				const bool is_selected = engine == current;
				if (ImGui::Selectable(name, is_selected))
					RayZath::Engine::Engine::instance().renderEngine(engine);

				if (is_selected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}
		
		ImGui::Text("direct light sampling");
		ImGui::Separator();

		// spot light		
		int spot_light_n = RayZath::Engine::Engine::instance().renderConfig().lightSampling().spotLight();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragInt("spot light", &spot_light_n, 0.1f, 0, 255, "%d"))
			RayZath::Engine::Engine::instance().renderConfig().lightSampling().spotLight(uint8_t(spot_light_n));
		// direct light		
		int direct_light_n = RayZath::Engine::Engine::instance().renderConfig().lightSampling().directLight();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragInt("direct light", &direct_light_n, 0.1f, 0, 255, "%d"))
			RayZath::Engine::Engine::instance().renderConfig().lightSampling().directLight(uint8_t(direct_light_n));


		ImGui::NewLine();
		ImGui::Text("tracing");
		ImGui::Separator();
		int max_depth = RayZath::Engine::Engine::instance().renderConfig().tracing().maxDepth();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragInt("max depth", &max_depth, 0.1f, 1, 255, "%d"))
			RayZath::Engine::Engine::instance().renderConfig().tracing().maxDepth(uint8_t(max_depth));
		int rpp = RayZath::Engine::Engine::instance().renderConfig().tracing().rpp();
		ImGui::SetNextItemWidth(left_width);
		if (ImGui::DragInt("rays per pixel", &rpp, 0.1f, 1, 1024, "%d"))
			RayZath::Engine::Engine::instance().renderConfig().tracing().rpp(uint8_t(rpp));

		ImGui::End();
	}
}