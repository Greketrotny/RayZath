#include "cpu_engine.hpp"

namespace RayZath::Engine::CPU
{
	void Engine::renderWorld(
		RayZath::Engine::World& hWorld,
		const RayZath::Engine::RenderConfig& render_config,
		[[maybe_unused]] const bool block,
		[[maybe_unused]] const bool sync)
	{
		m_engine_core.renderWorld(hWorld, render_config, block, sync);

		m_timing_string = std::string(m_engine_core.timeTable());
	}
	std::string Engine::timingsString()
	{
		return m_timing_string;
	}
}
