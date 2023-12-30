#include "cuda_engine.cuh"
#include "cuda_engine_core.cuh"

#include "point.h"

#include <locale>
#include <codecvt>
#include <string>

namespace RayZath::Cuda
{
	Engine::Engine()
		: m_engine_core(std::make_unique<EngineCore>())
	{}
	Engine::~Engine()
	{
		if (m_engine_core)
		{
			m_engine_core->renderer().terminateThread();
		}
	}

	void Engine::renderWorld(
		RayZath::Engine::World& hWorld,
		const RayZath::Engine::RenderConfig& render_config,
		const bool block,
		const bool sync)
	{
		m_engine_core->renderWorld(hWorld, render_config, block, sync);

		m_timing_string =
			"device:\n" + std::string(m_engine_core->renderTimeTable()) +
			"\nhost:\n" + std::string(m_engine_core->coreTimeTable());
	}

	std::string Engine::timingsString()
	{
		return m_timing_string;
	}
}