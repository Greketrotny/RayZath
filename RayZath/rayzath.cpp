#include "rayzath.h"

namespace RayZath
{
	Engine::Engine()
	{
		srand(time(NULL));
		mp_cuda_engine = new CudaEngine();
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
	void Engine::RenderWorld(RenderDevice device)
	{
		switch (device)
		{
			case RenderDevice::Default:
			case RenderDevice::CUDAGPU:
				mp_cuda_engine->RenderWorld(m_world);

			case RenderDevice::CPU:
				break;
		}
	}
}