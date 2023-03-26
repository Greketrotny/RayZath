#ifndef CUDA_ENGINE_H
#define CUDA_ENGINE_H

#include <string>
#include <memory>

namespace RayZath::Engine
{
	class World;
	struct RenderConfig;
}

namespace RayZath::Cuda
{
	class EngineCore;

	class Engine
	{
	private:
		std::unique_ptr<EngineCore> m_engine_core;
		std::string m_timing_string;


	public:
		Engine();
		~Engine();


	public:
		void renderWorld(
			RayZath::Engine::World& hWorld,
			const RayZath::Engine::RenderConfig& render_config,
			const bool block = true,
			const bool sync = true);
		std::string timingsString();
	};
}

#endif // CUDA_ENGINE_CORE_H
