#ifndef RAYZATH_H
#define RAYZATH_H

#include "world.h"
#include "engine_parts.h"

namespace RayZath
{
	namespace CudaEngine
	{
		class Engine;
	}

	class Engine
	{
		CudaEngine::Engine* mp_cuda_engine;
		World m_world;
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
		~Engine();


	public:
		static Engine& GetInstance();
		World& GetWorld();
		void RenderWorld(
			RenderDevice device = RenderDevice::Default,
			const bool block = true,
			const bool sync = true);

		std::wstring GetDebugInfo();
	};
}

#endif // !RAYZATH_H