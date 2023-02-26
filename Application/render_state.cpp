#include "render_state.hpp"

#include "rayzath.hpp"

#include "imgui.h"

namespace RayZath::UI::Windows
{
	void RenderState::open()
	{
		m_opened = true;
	}
	bool RenderState::isOpened() const
	{
		return m_opened;
	}

	void RenderState::update()
	{
		if (!m_opened) return;
		if (!ImGui::Begin("Rendering", &m_opened))
		{
			ImGui::End();
			return;
		}

		ImGui::TextWrapped("%s", RayZath::Engine::Engine::instance().debugInfo().c_str());

		ImGui::Text(
			"UI: %.3f ms/frame (%.1f FPS)",
			1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

		ImGui::End();
	}
}