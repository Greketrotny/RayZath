#include "cuda_engine.cuh"
#include "point.h"

#include <locale>
#include <codecvt>
#include <string>

namespace RayZath::Cuda
{
	Engine::~Engine()
	{
		m_engine_core.renderer().terminateThread();
	}

	void Engine::renderWorld(
		RayZath::Engine::World& hWorld,
		const RayZath::Engine::RenderConfig& render_config,
		const bool block,
		const bool sync)
	{
		m_engine_core.renderWorld(hWorld, render_config, block, sync);

		m_timing_string = "device:\n";
		m_timing_string += m_engine_core.renderTimeTable().toString(26);
		m_timing_string += "\nhost:\n";
		m_timing_string += m_engine_core.coreTimeTable().toString(26);
	}

	std::string Engine::timingsString()
	{
		return m_timing_string;
	}
}