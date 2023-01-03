#include "cpu_engine_core.hpp"

namespace RayZath::Engine::CPU
{
	EngineCore::EngineCore()
		: m_renderer(std::ref(*this))
	{}

	void EngineCore::renderWorld(
		RayZath::Engine::World& hWorld,
		const RayZath::Engine::RenderConfig& render_config,
		const bool block,
		const bool sync)
	{
		std::unique_lock lock(m_mtx);

		mp_world = &hWorld;
		m_renderer.render();
	}
	const RayZath::Engine::TimeTable& EngineCore::timeTable() const
	{
		return m_renderer.timeTable();
	}
}
