#include "cuda_engine.cuh"
#include "point.h"

#include <locale>
#include <codecvt>
#include <string>

namespace RayZath
{
	namespace CudaEngine
	{
		Engine::Engine()
		{

		}
		Engine::~Engine()
		{
			m_engine_core.GetRenderer().TerminateThread();
		}

		void Engine::RenderWorld(
			World& hWorld, 
			const RenderConfig& render_config, 
			const bool block, 
			const bool sync)
		{
			m_engine_core.RenderWorld(hWorld, render_config, block, sync);

			m_timing_string = "device:\n";
			m_timing_string += m_engine_core.GetRenderTimeTable().ToString(26);
			m_timing_string += "\nhost:\n";
			m_timing_string += m_engine_core.GetCoreTimeTable().ToString(26);
		}

		std::string Engine::GetTimingsString()
		{
			return m_timing_string;
		}
	}
}