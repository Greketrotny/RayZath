#ifndef CUDA_ENGINE_RENDERER
#define CUDA_ENGINE_RENDERER

#include "cuda_kernel_data.cuh"
#include "cuda_engine_parts.cuh"
#include "cuda_world.cuh"

#include "rzexception.hpp"
#include "cuda_exception.hpp"

#include <stdint.h>
#include <thread>

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

	template <size_t GC>
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
		void waitForEndOfAndClose(const size_t idx)
		{
			m_gates[idx].waitAndClose();
		}
		void waitForEndOf(const size_t idx)
		{
			m_gates[idx].wait();
		}
		void closeGate(const size_t& idx)
		{
			m_gates[idx].close();
		}
		void openGate(const size_t& idx)
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
		auto& checkGate(const size_t idx)
		{
			return m_gates[idx];
		}
	};

	struct TimeTable
	{
	private:
		RayZath::Engine::Timer m_timer, m_cycle_timer;
		std::vector<std::pair<std::string, float>> m_stamps;

	public:
		TimeTable();

		void appendStage(const std::string& s);
		void appendFullCycle(const std::string& s);
		void resetTable();
		void resetTime();
		std::string toString(const uint32_t width) const;
	};


	struct Renderer
	{
		enum class State
		{
			None,
			Idle,
			Work,
			wait
		};
		enum class Stage
		{
			None,
			Idle,
			Preprocess,
			MainRender,
			Postprocess
		};
	private:
		EngineCore* const mp_engine_core;

		std::unique_ptr<std::thread> mp_render_thread;
		std::atomic<bool> m_is_thread_alive, m_terminate_thread;
		RayZath::Engine::ThreadGate* mp_blocking_gate;

		std::unique_ptr<RayZath::Exception> m_exception;
		std::unique_ptr<RayZath::Cuda::Exception> m_cuda_exception;

		State m_state;
		Stage m_stage;
		FenceTrack<5> m_fence_track;

		TimeTable m_time_table;

		std::mutex m_mtx;


	public:
		Renderer(EngineCore* const engine_core);
		~Renderer();


	public:
		void launchThread();
		void terminateThread();


		FenceTrack<5>& fenceTrack();
		const TimeTable& timeTable() const;
		const State& state() const;
		const Stage& stage() const;

	private:
		void setState(const State& state);
		void setStage(const Stage& stage);

	private:
		void renderFunctionWrapper();
		void renderFunction() noexcept;
		bool checkTermination();

		void reportException(const RayZath::Exception& e);
		void reportCudaException(const RayZath::Cuda::Exception& e);
		void resetExceptions();
	public:
		void throwIfException();
	};
}

#endif 
