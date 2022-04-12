#include "cuda_engine_renderer.cuh"

#include "cuda_engine_core.cuh"

#include "cuda_preprocess_kernel.cuh"
#include "cuda_render_kernel.cuh"
#include "cuda_postprocess_kernel.cuh"

#include <algorithm>
#include <ios>
#include <sstream>

namespace RayZath::Cuda
{
	// ~~~~~~~~ [STRUCT] Indexer ~~~~~~~~
	Indexer::Indexer()
		: m_update_idx(0u)
		, m_render_idx(1u)
	{}

	const bool& Indexer::UpdateIdx() const
	{
		return m_update_idx;
	}
	const bool& Indexer::RenderIdx() const
	{
		return m_render_idx;
	}
	void Indexer::Swap()
	{
		std::swap(m_update_idx, m_render_idx);
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	// ~~~~~~~~ [STRUCT] TimeTable ~~~~~~~~
	TimeTable::TimeTable()
	{
		m_timer.Start();
	}

	void TimeTable::AppendStage(const std::string& s)
	{
		m_stamps.push_back({ s, m_timer.GetTime() });
	}
	void TimeTable::AppendFullCycle(const std::string& s)
	{
		m_stamps.push_back({ s, m_cycle_timer.GetTime() });
	}
	void TimeTable::ResetTime()
	{
		m_timer.Start();
		m_cycle_timer.Start();
	}
	void TimeTable::ResetTable()
	{
		m_stamps.clear();
	}
	std::string TimeTable::ToString(const uint32_t width) const
	{
		std::stringstream ss;
		for (auto& stamp : m_stamps)
		{
			ss.fill(' ');
			ss.width(width);
			ss << stamp.first << ": ";

			std::ignore = ss.width();
			ss.precision(3);
			ss << std::fixed << stamp.second << "ms\n";
		}
		return ss.str();
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	Renderer::Renderer(EngineCore* const engine_core)
		: mp_engine_core(engine_core)
		, m_is_thread_alive(false)
		, m_terminate_thread(false)
		, m_state(State::None)
		, m_stage(Stage::None)
		, mp_blocking_gate(nullptr)
		, m_fence_track(true)
	{}
	Renderer::~Renderer()
	{
		TerminateThread();
	}

	void Renderer::LaunchThread()
	{
		std::lock_guard<std::mutex> lg(m_mtx);
		if (!m_is_thread_alive)
		{
			m_terminate_thread = false;
			m_is_thread_alive = true;
			mp_blocking_gate = nullptr;
			ResetExceptions();

			if (mp_render_thread)
			{
				if (mp_render_thread->joinable())
					mp_render_thread->join();
			}
			mp_render_thread.reset(new std::thread(
				&Renderer::RenderFunctionWrapper,
				this));
		}
	}
	void Renderer::TerminateThread()
	{
		{
			std::lock_guard<std::mutex> lg(m_mtx);
			m_terminate_thread = true;
		}
		if (m_is_thread_alive)
		{
			mp_engine_core->GetFenceTrack().OpenAll();

			if (mp_render_thread->joinable())
				mp_render_thread->join();

			mp_render_thread.reset();
			m_is_thread_alive = false;
		}
	}

	FenceTrack<5>& Renderer::GetFenceTrack()
	{
		return m_fence_track;
	}
	const TimeTable& Renderer::GetTimeTable() const
	{
		return m_time_table;
	}
	const Renderer::State& Renderer::GetState() const
	{
		return m_state;
	}
	const Renderer::Stage& Renderer::GetStage() const
	{
		return m_stage;
	}

	void Renderer::SetState(const Renderer::State& state)
	{
		m_state = state;
	}
	void Renderer::SetStage(const Renderer::Stage& stage)
	{
		m_stage = stage;
	}

	void Renderer::RenderFunctionWrapper()
	{
		RenderFunction();
		std::lock_guard<std::mutex> lg(m_mtx);
		m_is_thread_alive = false;
		m_state = State::None;
		m_stage = Stage::None;
		m_fence_track.OpenAll();
	}
	void Renderer::RenderFunction() noexcept
	{
		m_time_table.ResetTime();
		try
		{
			while (!m_terminate_thread)
			{
				SetState(State::Idle);
				SetStage(Stage::Idle);

				// fine idling

				SetState(State::None);
				SetStage(Stage::None);
				m_fence_track.OpenGate(size_t(Renderer::Stage::Idle));
				mp_engine_core->GetFenceTrack().WaitForEndOfAndClose(size_t(EngineCore::Stage::Synchronization));
				if (CheckTermination()) return;
				m_time_table.ResetTable();
				m_time_table.AppendStage("wait for host");


				// Preprocess
				SetState(State::Work);
				SetStage(Stage::Preprocess);
				const auto& configs = mp_engine_core->GetLaunchConfigs(
					mp_engine_core->GetIndexer().RenderIdx()).GetConfigs();
				for (const auto& config : configs)
				{
					cudaSetDevice(config.GetDeviceId());

					if (config.GetUpdateFlag())
					{
						Kernel::PassReset
							<< <
							1u, 1u, 0u, mp_engine_core->GetRenderStream()
							>> >
							(mp_engine_core->GetCudaWorld(),
								config.GetCameraId());
						CudaErrorCheck(cudaStreamSynchronize(mp_engine_core->GetRenderStream()));
						CudaErrorCheck(cudaGetLastError());
					}
					m_time_table.AppendStage("camera buffer swap");

					if (config.GetUpdateFlag())
					{
						Kernel::GenerateCameraRay
							<< <
							config.GetGrid(),
							config.GetThreadBlock(),
							0u,
							mp_engine_core->GetRenderStream()
							>> > (
								mp_engine_core->GetGlobalKernel(mp_engine_core->GetIndexer().RenderIdx()),
								mp_engine_core->GetCudaWorld(),
								config.GetCameraId());
						CudaErrorCheck(cudaStreamSynchronize(mp_engine_core->GetRenderStream()));
						CudaErrorCheck(cudaGetLastError());
					}
					m_time_table.AppendStage("camera ray generation");
				}

				SetState(State::Wait);
				SetStage(Stage::None);
				m_fence_track.OpenGate(size_t(Stage::Preprocess));
				if (CheckTermination()) return;
				//mp_engine_core->GetFenceTrack().WaitForEndOfAndClose(?)


				// Main render
				SetState(State::Work);
				SetStage(Stage::MainRender);
				for (const auto& config : configs)
				{
					if (config.GetUpdateFlag())
					{
						Kernel::RenderFirstPass
							<< <
							config.GetGrid(),
							config.GetThreadBlock(),
							config.GetSharedMemorySize(),
							mp_engine_core->GetRenderStream()
							>> > (
								mp_engine_core->GetGlobalKernel(mp_engine_core->GetIndexer().RenderIdx()),
								mp_engine_core->GetCudaWorld(),
								config.GetCameraId());
						CudaErrorCheck(cudaStreamSynchronize(mp_engine_core->GetRenderStream()));
						CudaErrorCheck(cudaGetLastError());
						m_time_table.AppendStage("trace camera ray");

						Kernel::SpacialReprojection
							<< <
							config.GetGrid(),
							config.GetThreadBlock(),
							0u,
							mp_engine_core->GetRenderStream()
							>> > (
								mp_engine_core->GetCudaWorld(),
								config.GetCameraId());
						CudaErrorCheck(cudaStreamSynchronize(mp_engine_core->GetRenderStream()));
						CudaErrorCheck(cudaGetLastError());
						m_time_table.AppendStage("reprojection");

						Kernel::SegmentUpdate
							<< <
							1u, 1u, 0u, mp_engine_core->GetRenderStream()
							>> > (
								mp_engine_core->GetCudaWorld(),
								config.GetCameraId());
					}
					else
					{
						m_time_table.AppendStage("trace camera ray");
						m_time_table.AppendStage("reprojection");
					}

					const auto rpp = std::max(
						mp_engine_core->GetRenderConfig().GetTracing().GetRPP() - (config.GetUpdateFlag() ? 1 : 0),
						1);

					for (size_t i = 0; i < rpp; i++)
					{
						Kernel::RenderCumulativePass
							<< <
							config.GetGrid(),
							config.GetThreadBlock(),
							config.GetSharedMemorySize(),
							mp_engine_core->GetRenderStream()
							>> > (
								mp_engine_core->GetGlobalKernel(mp_engine_core->GetIndexer().RenderIdx()),
								mp_engine_core->GetCudaWorld(),
								config.GetCameraId());

						Kernel::SegmentUpdate
							<< <
							1u, 1u, 0u, mp_engine_core->GetRenderStream()
							>> > (
								mp_engine_core->GetCudaWorld(),
								config.GetCameraId());
						CudaErrorCheck(cudaStreamSynchronize(mp_engine_core->GetRenderStream()));
						CudaErrorCheck(cudaGetLastError());
					}
					m_time_table.AppendStage("trace cumulative");
				}

				SetState(State::Wait);
				SetStage(Stage::None);
				m_fence_track.OpenGate(size_t(Stage::MainRender));
				mp_engine_core->GetFenceTrack().WaitForEndOfAndClose(size_t(EngineCore::Stage::ResultTransfer));
				if (CheckTermination()) return;
				m_time_table.AppendStage("wait for result transfer");

				// Postprocess
				SetState(State::Work);
				SetStage(Stage::Postprocess);
				for (const auto& config : configs)
				{
					if (config.GetUpdateFlag() || true)
					{
						Kernel::FirstToneMap
							<< <
							config.GetGrid(),
							config.GetThreadBlock(),
							0u,
							mp_engine_core->GetRenderStream()
							>> > (
								mp_engine_core->GetCudaWorld(),
								config.GetCameraId());
					}
					else
					{
						Kernel::ToneMap
							<< <
							config.GetGrid(),
							config.GetThreadBlock(),
							0u,
							mp_engine_core->GetRenderStream()
							>> > (
								mp_engine_core->GetCudaWorld(),
								config.GetCameraId());
					}
					CudaErrorCheck(cudaStreamSynchronize(mp_engine_core->GetRenderStream()));
					CudaErrorCheck(cudaGetLastError());
					m_time_table.AppendStage("tone mapping");

					Kernel::PassUpdate
						<< <
						1u, 1u, 0u, mp_engine_core->GetRenderStream()
						>> > (
							mp_engine_core->GetCudaWorld(),
							config.GetCameraId());
					CudaErrorCheck(cudaStreamSynchronize(mp_engine_core->GetRenderStream()));
					CudaErrorCheck(cudaGetLastError());
				}

				m_time_table.AppendFullCycle("full render cycle");

				SetState(State::Idle);
				SetStage(Stage::Idle);
				m_fence_track.OpenGate(size_t(Stage::Postprocess));
			}
		}
		catch (const CudaException& e)
		{
			ReportCudaException(e);
		}
		catch (const Exception& e)
		{
			ReportException(e);
		}
		catch (...)
		{
			ReportException(Exception(
				"Rendering function unknown exception.",
				__FILE__, __LINE__));
		}
	}
	bool Renderer::CheckTermination()
	{
		return m_terminate_thread;
	}

	void Renderer::ReportException(const Exception& e)
	{
		if (!m_exception)
		{
			m_exception.reset(new Exception(e));
		}
	}
	void Renderer::ReportCudaException(const CudaException& e)
	{
		if (!m_cuda_exception)
		{
			m_cuda_exception.reset(new CudaException(e));
		}
	}
	void Renderer::ResetExceptions()
	{
		m_exception = nullptr;
		m_cuda_exception = nullptr;
	}
	void Renderer::ThrowIfException()
	{
		if (m_exception)
		{
			const Exception e = *m_exception;
			m_exception.reset();
			m_cuda_exception.reset();
			throw e;
		}

		if (m_cuda_exception)
		{
			const CudaException e = *m_cuda_exception;
			m_exception.reset();
			m_cuda_exception.reset();
			throw e;
		}
	}
}