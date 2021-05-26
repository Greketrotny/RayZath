#include "rayzath.h"
#include "cuda_engine.cuh"

namespace RayZath
{
	Engine::Engine()
	{
		srand(unsigned int(time(NULL)));
		mp_cuda_engine = new CudaEngine::Engine();
	}
	Engine::~Engine()
	{
		if (mp_cuda_engine) delete mp_cuda_engine;
	}

	Engine& Engine::GetInstance()
	{
		static Engine rayzeth_instance;
		return rayzeth_instance;
	}
	World& Engine::GetWorld()
	{
		return m_world;
	}
	void Engine::RenderWorld(
		RenderDevice device,
		const bool block,
		const bool sync)
	{
		switch (device)
		{
			case RenderDevice::Default:
			case RenderDevice::CUDAGPU:
				mp_cuda_engine->RenderWorld(m_world, block, sync);
				break;

			case RenderDevice::CPU:
				break;
		}
	}

	std::wstring Engine::GetDebugInfo()
	{
		return mp_cuda_engine->GetTimingsString();
	}
}