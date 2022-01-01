#ifndef RAYZATH_H
#define RAYZATH_H

#include "world.h"
#include "loader.h"
#include "engine_parts.h"

namespace RayZath::Cuda
{
	class Engine;
}

namespace RayZath::Engine
{
	class Engine
	{
	private:
		std::unique_ptr<RayZath::Cuda::Engine> m_cuda_engine;
		std::unique_ptr<World> m_world;

		RenderConfig m_render_config;

	public:
		enum class RenderDevice
		{
			Default,
			CPU,
			CUDAGPU
		};


	private:
		Engine();
	public:
		Engine(const Engine&) = delete;
		Engine& operator=(const Engine&) = delete;


	public:
		static Engine& GetInstance();
		World& GetWorld();
		RenderConfig& GetRenderConfig();

		void RenderWorld(
			RenderDevice device = RenderDevice::Default,
			const bool block = true,
			const bool sync = true);

		std::string GetDebugInfo();
	};
}

#endif // !RAYZATH_H