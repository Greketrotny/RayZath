#ifndef RAYZATH_H
#define RAYZATH_H

#include "world.hpp"
#include "engine_parts.hpp"
#include "loader.hpp"
#include "saver.hpp"

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
		static Engine& instance();
		World& world();
		RenderConfig& renderConfig();

		void renderWorld(
			RenderDevice device = RenderDevice::Default,
			const bool block = true,
			const bool sync = true);

		std::string debugInfo();
	};
}

#endif // !RAYZATH_H