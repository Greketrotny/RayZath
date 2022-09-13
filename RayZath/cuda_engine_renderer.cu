#include "cuda_engine_renderer.cuh"

#include "cuda_engine_core.cuh"
#include "cuda_exception.hpp"

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

	const bool& Indexer::updateIdx() const
	{
		return m_update_idx;
	}
	const bool& Indexer::renderIdx() const
	{
		return m_render_idx;
	}
	void Indexer::swap()
	{
		std::swap(m_update_idx, m_render_idx);
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	// ~~~~~~~~ [STRUCT] TimeTable ~~~~~~~~
	TimeTable::TimeTable()
	{
		m_timer.start();
	}

	void TimeTable::appendStage(const std::string& s)
	{
		m_stamps.push_back({s, m_timer.time()});
	}
	void TimeTable::appendFullCycle(const std::string& s)
	{
		m_stamps.push_back({s, m_cycle_timer.time()});
	}
	void TimeTable::resetTime()
	{
		m_timer.start();
		m_cycle_timer.start();
	}
	void TimeTable::resetTable()
	{
		m_stamps.clear();
	}
	std::string TimeTable::toString(const uint32_t width) const
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
		terminateThread();
	}

	void Renderer::launchThread()
	{
		std::lock_guard<std::mutex> lg(m_mtx);
		if (!m_is_thread_alive)
		{
			m_terminate_thread = false;
			m_is_thread_alive = true;
			mp_blocking_gate = nullptr;
			resetExceptions();

			if (mp_render_thread)
			{
				if (mp_render_thread->joinable())
					mp_render_thread->join();
			}
			mp_render_thread.reset(new std::thread(
				&Renderer::renderFunctionWrapper,
				this));
		}
	}
	void Renderer::terminateThread()
	{
		{
			std::lock_guard<std::mutex> lg(m_mtx);
			m_terminate_thread = true;
		}
		if (m_is_thread_alive)
		{
			mp_engine_core->fenceTrack().openAll();

			if (mp_render_thread->joinable())
				mp_render_thread->join();

			mp_render_thread.reset();
			m_is_thread_alive = false;
		}
	}

	FenceTrack<5>& Renderer::fenceTrack()
	{
		return m_fence_track;
	}
	const TimeTable& Renderer::timeTable() const
	{
		return m_time_table;
	}
	const Renderer::State& Renderer::state() const
	{
		return m_state;
	}
	const Renderer::Stage& Renderer::stage() const
	{
		return m_stage;
	}

	void Renderer::setState(const Renderer::State& state)
	{
		m_state = state;
	}
	void Renderer::setStage(const Renderer::Stage& stage)
	{
		m_stage = stage;
	}

	void Renderer::renderFunctionWrapper()
	{
		renderFunction();
		std::lock_guard<std::mutex> lg(m_mtx);
		m_is_thread_alive = false;
		m_state = State::None;
		m_stage = Stage::None;
		m_fence_track.openAll();
	}
	void Renderer::renderFunction() noexcept
	{
		m_time_table.resetTime();
		try
		{
			while (!m_terminate_thread)
			{
				setState(State::Idle);
				setStage(Stage::Idle);

				// fine idling

				setState(State::None);
				setStage(Stage::None);
				m_fence_track.openGate(size_t(Renderer::Stage::Idle));
				mp_engine_core->fenceTrack().waitForEndOfAndClose(size_t(EngineCore::Stage::Synchronization));
				if (checkTermination()) return;
				m_time_table.resetTable();
				m_time_table.appendStage("wait for host");


				// Preprocess
				setState(State::Work);
				setStage(Stage::Preprocess);
				const auto& configs = mp_engine_core->launchConfigs(
					mp_engine_core->indexer().renderIdx()).GetConfigs();
				for (const auto& config : configs)
				{
					cudaSetDevice(config.GetDeviceId());

					if (config.GetUpdateFlag())
					{
						Kernel::passReset
							<< <
							1u, 1u, 0u, mp_engine_core->renderStream()
							>> >
							(mp_engine_core->cudaWorld(),
								config.GetCameraId());
						RZAssertCoreCUDA(cudaStreamSynchronize(mp_engine_core->renderStream()));
						RZAssertCoreCUDA(cudaGetLastError());
					}
					m_time_table.appendStage("camera buffer swap");

					if (config.GetUpdateFlag())
					{
						Kernel::generateCameraRay
							<< <
							config.GetGrid(),
							config.GetThreadBlock(),
							0u,
							mp_engine_core->renderStream()
							>> > (
								mp_engine_core->globalKernel(mp_engine_core->indexer().renderIdx()),
								mp_engine_core->cudaWorld(),
								config.GetCameraId());
						RZAssertCoreCUDA(cudaStreamSynchronize(mp_engine_core->renderStream()));
						RZAssertCoreCUDA(cudaGetLastError());
					}
					m_time_table.appendStage("camera ray generation");
				}

				setState(State::wait);
				setStage(Stage::None);
				m_fence_track.openGate(size_t(Stage::Preprocess));
				if (checkTermination()) return;
				//mp_engine_core->GetFenceTrack().WaitForEndOfAndClose(?)


				// Main render
				setState(State::Work);
				setStage(Stage::MainRender);
				for (const auto& config : configs)
				{
					if (config.GetUpdateFlag())
					{
						Kernel::renderFirstPass
							<< <
							config.GetGrid(),
							config.GetThreadBlock(),
							config.GetSharedMemorySize(),
							mp_engine_core->renderStream()
							>> > (
								mp_engine_core->globalKernel(mp_engine_core->indexer().renderIdx()),
								mp_engine_core->cudaWorld(),
								config.GetCameraId());
						RZAssertCoreCUDA(cudaStreamSynchronize(mp_engine_core->renderStream()));
						RZAssertCoreCUDA(cudaGetLastError());
						m_time_table.appendStage("trace camera ray");

						Kernel::spacialReprojection
							<< <
							config.GetGrid(),
							config.GetThreadBlock(),
							0u,
							mp_engine_core->renderStream()
							>> > (
								mp_engine_core->cudaWorld(),
								config.GetCameraId());
						RZAssertCoreCUDA(cudaStreamSynchronize(mp_engine_core->renderStream()));
						RZAssertCoreCUDA(cudaGetLastError());
						m_time_table.appendStage("reprojection");

						Kernel::segmentUpdate
							<< <
							1u, 1u, 0u, mp_engine_core->renderStream()
							>> > (
								mp_engine_core->cudaWorld(),
								config.GetCameraId());
					}
					else
					{
						m_time_table.appendStage("trace camera ray");
						m_time_table.appendStage("reprojection");
					}

					const auto rpp = std::max(
						mp_engine_core->renderConfig().tracing().rpp() - (config.GetUpdateFlag() ? 1 : 0),
						1);

					for (size_t i = 0; i < rpp; i++)
					{
						Kernel::renderCumulativePass
							<< <
							config.GetGrid(),
							config.GetThreadBlock(),
							config.GetSharedMemorySize(),
							mp_engine_core->renderStream()
							>> > (
								mp_engine_core->globalKernel(mp_engine_core->indexer().renderIdx()),
								mp_engine_core->cudaWorld(),
								config.GetCameraId());

						Kernel::segmentUpdate
							<< <
							1u, 1u, 0u, mp_engine_core->renderStream()
							>> > (
								mp_engine_core->cudaWorld(),
								config.GetCameraId());
						RZAssertCoreCUDA(cudaStreamSynchronize(mp_engine_core->renderStream()));
						RZAssertCoreCUDA(cudaGetLastError());
					}
					m_time_table.appendStage("trace cumulative");
				}

				setState(State::wait);
				setStage(Stage::None);
				m_fence_track.openGate(size_t(Stage::MainRender));
				mp_engine_core->fenceTrack().waitForEndOfAndClose(size_t(EngineCore::Stage::ResultTransfer));
				if (checkTermination()) return;
				m_time_table.appendStage("wait for result transfer");

				// Postprocess
				setState(State::Work);
				setStage(Stage::Postprocess);
				for (const auto& config : configs)
				{
					if (config.GetUpdateFlag() || true)
					{
						Kernel::firstToneMap
							<< <
							config.GetGrid(),
							config.GetThreadBlock(),
							0u,
							mp_engine_core->renderStream()
							>> > (
								mp_engine_core->cudaWorld(),
								config.GetCameraId());
					}
					else
					{
						Kernel::toneMap
							<< <
							config.GetGrid(),
							config.GetThreadBlock(),
							0u,
							mp_engine_core->renderStream()
							>> > (
								mp_engine_core->cudaWorld(),
								config.GetCameraId());
					}
					RZAssertCoreCUDA(cudaStreamSynchronize(mp_engine_core->renderStream()));
					RZAssertCoreCUDA(cudaGetLastError());
					m_time_table.appendStage("tone mapping");

					Kernel::passUpdate
						<< <
						1u, 1u, 0u, mp_engine_core->renderStream()
						>> > (
							mp_engine_core->cudaWorld(),
							config.GetCameraId());
					RZAssertCoreCUDA(cudaStreamSynchronize(mp_engine_core->renderStream()));
					RZAssertCoreCUDA(cudaGetLastError());

					Kernel::rayCast
						<< <
						1u, 1u, 0u, mp_engine_core->renderStream()
						>> > (mp_engine_core->cudaWorld(),
							config.GetCameraId());
					RZAssertCoreCUDA(cudaStreamSynchronize(mp_engine_core->renderStream()));
					RZAssertCoreCUDA(cudaGetLastError());
				}

				m_time_table.appendFullCycle("full render cycle");

				setState(State::Idle);
				setStage(Stage::Idle);
				m_fence_track.openGate(size_t(Stage::Postprocess));
			}
		}
		catch (const Cuda::Exception& e)
		{
			reportCudaException(e);
		}
		catch (const RayZath::Exception& e)
		{
			reportException(e);
		}
		catch (...)
		{
			reportException(Cuda::Exception(
				"Rendering function unknown exception.",
				__FILE__, __LINE__));
		}
	}
	bool Renderer::checkTermination()
	{
		return m_terminate_thread;
	}

	void Renderer::reportException(const RayZath::Exception& e)
	{
		if (!m_exception)
		{
			m_exception.reset(new RayZath::Exception(e));
		}
	}
	void Renderer::reportCudaException(const Cuda::Exception& e)
	{
		if (!m_cuda_exception)
		{
			m_cuda_exception.reset(new Cuda::Exception(e));
		}
	}
	void Renderer::resetExceptions()
	{
		m_exception = nullptr;
		m_cuda_exception = nullptr;
	}
	void Renderer::throwIfException()
	{
		if (m_exception)
		{
			const RayZath::Exception e = *m_exception;
			m_exception.reset();
			m_cuda_exception.reset();
			throw e;
		}

		if (m_cuda_exception)
		{
			const RayZath::Cuda::Exception e = *m_cuda_exception;
			m_exception.reset();
			m_cuda_exception.reset();
			throw e;
		}
	}
}