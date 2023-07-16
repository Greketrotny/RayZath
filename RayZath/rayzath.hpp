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
namespace RayZath::Engine::CPU
{
	class Engine;
}

namespace RayZath::Engine
{
	class Engine
	{
	public:
		enum class RenderEngine
		{
			CPU,
			CUDAGPU
		};
		static const std::map<RenderEngine, std::string_view> engine_name;
	private:
		std::unique_ptr<World> m_world;
		std::unique_ptr<RayZath::Cuda::Engine> m_cuda_engine;
		std::unique_ptr<RayZath::Engine::CPU::Engine> m_cpu_engine;

		RenderConfig m_render_config;
		RenderEngine m_render_engine;

	private:
		Engine();
	public:
		Engine(const Engine&) = delete;
		Engine& operator=(const Engine&) = delete;


	public:
		static Engine& instance();
		RenderConfig& renderConfig();
		RenderEngine renderEngine() const;
		void renderEngine(RenderEngine engine);

		void renderWorld(
			RenderEngine device,
			const bool block = true,
			const bool sync = true);
		void renderWorld(
			const bool block = true,
			const bool sync = true);


		std::string debugInfo();
	};
}

#endif // !RAYZATH_H
