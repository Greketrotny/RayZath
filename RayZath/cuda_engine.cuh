#ifndef CUDA_ENGINE_H
#define CUDA_ENGINE_H

#include "cuda_engine_core.cuh"

#include "rzexception.h"

#include <thread>
#include <condition_variable>
#include <mutex>
#include <chrono>

#include <string>
#include <sstream>
#include <iomanip>
#include <vector>

namespace RayZath
{
	namespace CudaEngine
	{
		class Engine
		{
		private:
			CudaEngineCore m_engine_core;
			std::string m_timing_string;


		public:
			Engine();
			~Engine();


		public:
			void RenderWorld(
				World& hWorld,
				const RenderConfig& render_config,
				const bool block = true, 
				const bool sync = true);
			std::string GetTimingsString();
		};
	}
}

#endif // CUDA_ENGINE_CORE_H