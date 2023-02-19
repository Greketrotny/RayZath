#include "rayzath.hpp"

#include "cuda_engine.cuh"
#include "cpu_engine.hpp"

namespace RayZath::Engine
{
	Engine::Engine()
		: m_world(std::make_unique<World>())
		, m_cuda_engine(std::make_unique<RayZath::Cuda::Engine>())
		, m_cpu_engine(std::make_unique<RayZath::Engine::CPU::Engine>())
		, m_render_engine(RenderEngine::CUDAGPU)
	{
		srand((unsigned int)(time(NULL)));
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
	Engine::RenderEngine Engine::renderEngine() const
	{
		return m_render_engine;
	}
	void Engine::renderEngine(RenderEngine engine)
	{
		m_render_engine = engine;
	}

	void Engine::renderWorld(
		RenderEngine engine,
		const bool block,
		const bool sync)
	{
		switch (engine)
		{
			case RenderEngine::CUDAGPU:
				m_cuda_engine->renderWorld(*m_world, m_render_config, block, sync);
				break;
			case RenderEngine::CPU:
				m_cpu_engine->renderWorld(*m_world, m_render_config, block, sync);
				break;
			default:
				RZThrowCore("unsupported RenderEngine type");
		}
	}
	void Engine::renderWorld(const bool block, const bool sync)
	{
		renderWorld(m_render_engine, block, sync);
	}

	int Engine::renderWorld(
		const std::filesystem::path& scene_path, 
		const std::filesystem::path& report_path, 
		const std::filesystem::path& config_path)
	{
		m_world->loader().loadScene(scene_path);
		for (size_t i = 0; i < 1000; i++)
		{
			renderWorld(true, true);
		}

		auto& cameras = m_world->container<ObjectType::Camera>();
		if (cameras.count())
		{
			m_world->saver().saveMap<ObjectType::Texture>(
				cameras[0]->imageBuffer(), report_path, "render");
		}

		return 0;
	}
	

	std::string Engine::debugInfo()
	{
		switch (m_render_engine)
		{
			case RenderEngine::CUDAGPU:
				return m_cuda_engine->timingsString();
			case RenderEngine::CPU:
				return m_cpu_engine->timingsString();
			default:
				RZThrowCore("unsupported RenderEngine type");
		}
	}
}