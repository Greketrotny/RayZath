#include "rayzath.hpp"
#include "cuda_engine.cuh"

namespace RayZath::Engine
{
	Engine::Engine()
		: m_world(std::make_unique<World>())
		, m_cuda_engine(std::make_unique<RayZath::Cuda::Engine>())
	{
		srand(unsigned int(time(NULL)));
	}

	Engine& Engine::instance()
	{
		static Engine engine_instance;
		return engine_instance;
	}
	World& Engine::world()
	{
		return *m_world;
	}
	RenderConfig& Engine::renderConfig()
	{
		return m_render_config;
	}

	void Engine::renderWorld(
		RenderDevice device,
		const bool block,
		const bool sync)
	{
		switch (device)
		{
			case RenderDevice::Default:
			case RenderDevice::CUDAGPU:
				m_cuda_engine->renderWorld(*m_world, m_render_config, block, sync);
				break;

			case RenderDevice::CPU:
				break;
		}
	}

	std::string Engine::debugInfo()
	{
		return m_cuda_engine->timingsString();
	}
}