#ifndef RAYZATH_H
#define RAYZATH_H

#include "cuda_engine.cuh"

#include "world.h"

namespace RayZath
{
	class Engine
	{
	//private:
	public:
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
		void RenderWorld(RenderDevice device = RenderDevice::Default);
	};
}

#endif // !RAYZATH_H