#ifndef CUDA_ENGINE_RENDERER
#define CUDA_ENGINE_RENDERER

#include "cuda_kernel_data.cuh"
#include "cuda_engine_parts.cuh"
#include "cuda_world.cuh"

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
		const bool& UpdateIdx() const;
		const bool& RenderIdx() const;
		void Swap();
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
					g.Open();
				}
			}
			else
			{
				for (auto& g : m_gates)
				{
					g.Close();
				}
			}
		}
		~FenceTrack()
		{
			for (auto& g : m_gates)
			{
				g.Open();
			}
		}


	public:
		void WaitForEndOfAndClose(const size_t idx)
		{
			m_gates[idx].WaitAndClose();
		}
		void WaitForEndOf(const size_t idx)
		{
			m_gates[idx].Wait();
		}
		void CloseGate(const size_t& idx)
		{
			m_gates[idx].Close();
		}
		void OpenGate(const size_t& idx)
		{
			m_gates[idx].Open();
		}
		void OpenAll()
		{
			for (auto& g : m_gates)
			{
				g.Open();
			}
		}
	};

	struct TimeTable
	{
	private:
		RayZath::Engine::Timer m_timer, m_cycle_timer;
		std::vector<std::pair<std::string, float>> m_stamps;

	public:
		TimeTable();

	public:
		void AppendStage(const std::string& s);
		void AppendFullCycle(const std::string& s);
		void ResetTable();
		void ResetTime();
		std::string ToString(const uint32_t width) const;
	};


	struct Renderer
	{
		enum class State
		{
			None,
			Idle,
			Work,
			Wait
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

		std::unique_ptr<Exception> m_exception;
		std::unique_ptr<CudaException> m_cuda_exception;

		State m_state;
		Stage m_stage;
		FenceTrack<5> m_fence_track;

		TimeTable m_time_table;

		std::mutex m_mtx;


	public:
		Renderer(EngineCore* const engine_core);
		~Renderer();


	public:
		void LaunchThread();
		void TerminateThread();


		FenceTrack<5>& GetFenceTrack();
		const TimeTable& GetTimeTable() const;
		const State& GetState() const;
		const Stage& GetStage() const;

	private:
		void SetState(const State& state);
		void SetStage(const Stage& stage);

	private:
		void RenderFunctionWrapper();
		void RenderFunction() noexcept;
		bool CheckTermination();

		void ReportException(const Exception& e);
		void ReportCudaException(const CudaException& e);
		void ResetExceptions();
	public:
		void ThrowIfException();
	};
}

#endif 