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

		void Engine::RenderWorld(World& hWorld, const bool block, const bool sync)
		{
			m_engine_core.RenderWorld(hWorld, block, sync);

			std::string s1 = m_engine_core.GetRenderTimeTable().ToString(25);
			std::string s2 = m_engine_core.GetCoreTimeTable().ToString(25);

			m_timing_string = L"device:\n";
			for (auto& l : s1)
			{
				m_timing_string += wchar_t(l);
			}
			m_timing_string += L"\nhost:\n";
			for (auto& l : s2)
			{
				m_timing_string += wchar_t(l);
			}
		}

		std::wstring Engine::GetTimingsString()
		{
			return m_timing_string;
		}
	}
}