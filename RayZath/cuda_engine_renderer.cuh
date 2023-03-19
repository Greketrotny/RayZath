#ifndef CUDA_ENGINE_RENDERER
#define CUDA_ENGINE_RENDERER

#include "cuda_kernel_data.cuh"
#include "cuda_engine_parts.cuh"
#include "cuda_world.cuh"

#include "engine_parts.hpp"

#include "rzexception.hpp"
#include "cuda_exception.hpp"

#include <stdint.h>
#include <thread>
#include <iostream>

namespace RayZath::Cuda
{
	class EngineCore;
	struct Renderer;

	struct Indexer
	{
	private:
		bool m_update_idx, m_render_idx;

	public:
		Indexer();

	public:
		const bool& updateIdx() const;
		const bool& renderIdx() const;
		void swap();
	};

	template <std::size_t GC>
	struct FenceTrack
	{
	private:
		std::array<RayZath::Engine::ThreadGate, GC> m_gates;

	public:
		FenceTrack(const bool all_opened)
		{
			if (all_opened)
			{
				for (auto& g : m_gates)
				{
					g.open();
				}
			}
			else
			{
				for (auto& g : m_gates)
				{
					g.close();
				}
			}
		}
		~FenceTrack()
		{
			for (auto& g : m_gates)
			{
				g.open();
			}
		}


	public:
		RayZath::Engine::Timer::duration_t waitFor(const std::size_t idx)
		{
			RayZath::Engine::Timer timer;
			m_gates[idx].waitAndClose();
			return timer.peek();
		}
		void waitForKeepOpen(const std::size_t idx)
		{
			m_gates[idx].wait();
		}
		void closeGate(const std::size_t& idx)
		{
			m_gates[idx].close();
		}
		void openGate(const std::size_t& idx)
		{
			m_gates[idx].open();
		}
		void openAll()
		{
			for (auto& g : m_gates)
			{
				g.open();
			}
		}
		auto& checkGate(const std::size_t idx)
		{
			return m_gates[idx];
		}
	};

	struct Renderer
	{
	public:
		enum class Stage
		{
			Idle,
			Preprocess,
			MainRender,
			Postprocess
		};
	private:
		EngineCore* const mp_engine_core;

		std::thread m_render_thread;
		std::atomic<bool> m_terminate_render_thread;

		std::unique_ptr<RayZath::Exception> m_exception;
		std::unique_ptr<RayZath::Cuda::Exception> m_cuda_exception;

		FenceTrack<5> m_fence_track;

		RayZath::Engine::TimeTable m_time_table;


	public:
		Renderer(EngineCore* engine_core);
		~Renderer();


		void launchThread();
		void terminateThread();

		FenceTrack<5>& fenceTrack();
		const RayZath::Engine::TimeTable& timeTable() const { return m_time_table; };
	private:
		void renderFunctionWrapper();
		void renderFunction() noexcept;
		bool shouldReturn();

		void reportException(const RayZath::Exception& e);
		void reportCudaException(const RayZath::Cuda::Exception& e);
		void resetExceptions();
	public:
		void throwIfException();
	};
}

#endif 
