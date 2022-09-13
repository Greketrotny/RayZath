#ifndef CUDA_ENGINE_H
#define CUDA_ENGINE_H

#include "cuda_engine_core.cuh"

#include <thread>
#include <condition_variable>
#include <mutex>
#include <chrono>

#include <string>
#include <sstream>
#include <iomanip>
#include <vector>

namespace RayZath::Cuda
{
	class Engine
	{
	private:
		EngineCore m_engine_core;
		std::string m_timing_string;


	public:
		Engine() = default;
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