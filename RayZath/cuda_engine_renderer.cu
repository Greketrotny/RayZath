#include "cuda_engine_renderer.cuh"

#include "cuda_engine_core.cuh"
#include "cuda_exception.hpp"

#include "cuda_preprocess_kernel.cuh"
#include "cuda_render_kernel.cuh"
#include "cuda_postprocess_kernel.cuh"

namespace RayZath::Cuda
{
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


	Renderer::Renderer(EngineCore* engine_core)
		: mp_engine_core(engine_core)
		, m_terminate_render_thread(false)
		, m_fence_track(true)
	{}
	Renderer::~Renderer()
	{
		terminateThread();
	}

	void Renderer::launchThread()
	{
		if (!m_render_thread.joinable())
		{
			m_terminate_render_thread = false;
			resetExceptions();
			m_render_thread = std::thread(
				&Renderer::renderFunctionWrapper,
				this);
		}
	}
	void Renderer::terminateThread()
	{
		if (m_render_thread.joinable())
		{
			m_terminate_render_thread = true;
			mp_engine_core->fenceTrack().openAll();
			m_render_thread.join();
		}
	}

	FenceTrack<5>& Renderer::fenceTrack()
	{
		return m_fence_track;
	}
	
	void Renderer::renderFunctionWrapper()
	{
		renderFunction();
		m_terminate_render_thread = true;
		m_fence_track.openAll();
	}
	void Renderer::renderFunction() noexcept
	{
		try
		{
			while (!shouldReturn())
			{
				m_time_table.setWaitTime(
					"generate camera ray", 
					mp_engine_core->fenceTrack().waitFor(std::size_t(EngineCore::Stage::Synchronization)));
				if (shouldReturn()) return;

				// Preprocess
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
					m_time_table.update("generate camera ray");
				}

				m_fence_track.openGate(std::size_t(Stage::Preprocess));
				if (shouldReturn()) return;


				// Main render
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
						m_time_table.update("trace camera ray");

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
						m_time_table.update("temporal reproject");

						Kernel::segmentUpdate
							<< <
							1u, 1u, 0u, mp_engine_core->renderStream()
							>> > (
								mp_engine_core->cudaWorld(),
								config.GetCameraId());
					}
					else
					{
						m_time_table.update("trace camera ray");
						m_time_table.update("temporal reproject");
					}

					for (std::size_t i = std::size_t(config.GetUpdateFlag());
						i < mp_engine_core->renderConfig().tracing().rpp();
						i++)
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
					}

					Kernel::rayCast
						<< <
						1u, 1u, 0u, mp_engine_core->renderStream()
						>> > (mp_engine_core->cudaWorld(),
							config.GetCameraId());
					RZAssertCoreCUDA(cudaStreamSynchronize(mp_engine_core->renderStream()));
					RZAssertCoreCUDA(cudaGetLastError());

					m_time_table.update("trace cumulative");
				}

				m_fence_track.openGate(std::size_t(Stage::MainRender));
				m_time_table.setWaitTime(
					"tone mapping", 
					mp_engine_core->fenceTrack().waitFor(std::size_t(EngineCore::Stage::ResultTransfer)));
				if (shouldReturn()) return;

				// Postprocess
				for (const auto& config : configs)
				{
					if (config.GetUpdateFlag())
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

					Kernel::passUpdate
						<< <
						1u, 1u, 0u, mp_engine_core->renderStream()
						>> > (
							mp_engine_core->cudaWorld(),
							config.GetCameraId());
					RZAssertCoreCUDA(cudaStreamSynchronize(mp_engine_core->renderStream()));
					RZAssertCoreCUDA(cudaGetLastError());
					m_time_table.update("tone mapping");
				}

				m_time_table.updateCycle("full cycle");
				m_fence_track.openGate(std::size_t(Stage::Postprocess));
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
	bool Renderer::shouldReturn()
	{
		return m_terminate_render_thread;
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