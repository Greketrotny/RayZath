#include "engine.hpp"

namespace RayZath::CPU
{
	void Engine::renderWorld(
		RayZath::Engine::World& hWorld,
		const RayZath::Engine::RenderConfig& render_config,
		[[maybe_unused]] const bool block,
		[[maybe_unused]] const bool sync)
	{
		m_engine_core.renderWorld(hWorld, render_config, block, sync);
	}
}
