#include "cuda_engine_core.cuh"

#include "cuda_preprocess_kernel.cuh"
#include "cuda_render_kernel.cuh"
#include "cuda_postprocess_kernel.cuh"

#include "point.h"

#include <algorithm>

namespace RayZath
{
	namespace CudaEngine
	{
		// ~~~~~~~~ [STRUCT] CudaIndexer ~~~~~~~~
		CudaIndexer::CudaIndexer()
			: m_update_idx(0u)
			, m_render_idx(1u)
		{}

		const bool& CudaIndexer::UpdateIdx() const
		{
			return m_update_idx;
		}
		const bool& CudaIndexer::RenderIdx() const
		{
			return m_render_idx;
		}
		void CudaIndexer::Swap()
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

				ss.width();
				ss.precision(3);
				ss << std::fixed << stamp.second << "ms\n";
			}
			return ss.str();
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


		// ~~~~~~~~ [STRUCT] CudaRenderer ~~~~~~~~
		CudaRenderer::CudaRenderer(CudaEngineCore* const engine_core)
			: mp_engine_core(engine_core)
			, m_is_thread_alive(false)
			, m_terminate_thread(false)
			, m_state(State::None)
			, m_stage(Stage::None)
			, mp_blocking_gate(nullptr)
			, m_fence_track(true)
		{}
		CudaRenderer::~CudaRenderer()
		{
			TerminateThread();
		}

		void CudaRenderer::LaunchThread()
		{
			std::lock_guard<std::mutex> lg(m_mtx);
			if (!m_is_thread_alive)
			{
				m_terminate_thread = false;
				m_is_thread_alive = true;
				mp_blocking_gate = nullptr;
				ResetExceptions();
				mp_render_thread = std::make_unique<std::thread>(
					&CudaRenderer::RenderFunctionWrapper, 
					this);
			}			
		}
		void CudaRenderer::TerminateThread()
		{
			{
				std::lock_guard<std::mutex> lg(m_mtx);
				m_terminate_thread = true;
			}
			if (m_is_thread_alive)
			{
				mp_engine_core->GetFenceTrack().OpenAll();
				//if (mp_blocking_gate)
				//	mp_blocking_gate->Open();

				if (mp_render_thread->joinable())
					mp_render_thread->join();

				m_is_thread_alive = false;
				mp_render_thread = nullptr;
			}
		}

		FenceTrack<5>& CudaRenderer::GetFenceTrack()
		{
			return m_fence_track;
		}
		const TimeTable& CudaRenderer::GetTimeTable() const
		{
			return m_time_table;
		}
		const CudaRenderer::State& CudaRenderer::GetState() const
		{
			return m_state;
		}
		const CudaRenderer::Stage& CudaRenderer::GetStage() const
		{
			return m_stage;
		}

		void CudaRenderer::SetState(const CudaRenderer::State& state)
		{
			m_state = state;
		}
		void CudaRenderer::SetStage(const CudaRenderer::Stage& stage)
		{
			m_stage = stage;
		}

		void CudaRenderer::RenderFunctionWrapper()
		{
			RenderFunction();
			std::lock_guard<std::mutex> lg(m_mtx);
			m_is_thread_alive = false;
			m_state = State::None;
			m_stage = Stage::None;
			m_fence_track.OpenAll();
		}
		void CudaRenderer::RenderFunction() noexcept
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
					m_fence_track.OpenGate(size_t(CudaRenderer::Stage::Idle));
					mp_engine_core->GetFenceTrack().WaitForEndOf(size_t(CudaEngineCore::Stage::CameraReconstruction));
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
							CudaKernel::DepthBufferReset
								<< <
								config.GetGrid(),
								config.GetThreadBlock(),
								0u,
								mp_engine_core->GetRenderStream()
								>> >
								(mp_engine_core->GetCudaWorld(), config.GetCameraId());
							CudaErrorCheck(cudaStreamSynchronize(mp_engine_core->GetRenderStream()));
							CudaErrorCheck(cudaGetLastError());
						}
						m_time_table.AppendStage("buffer reset");

						CudaKernel::CudaCameraUpdateSamplesNumber
							<< <
							1u, 1u, 0u, mp_engine_core->GetRenderStream()
							>> >
							(mp_engine_core->GetCudaWorld(),
								config.GetCameraId(),
								config.GetUpdateFlag());
						CudaErrorCheck(cudaStreamSynchronize(mp_engine_core->GetRenderStream()));
						CudaErrorCheck(cudaGetLastError());
						m_time_table.AppendStage("sample update");
					}

					SetState(State::Wait);
					SetStage(Stage::None);
					m_fence_track.OpenGate(size_t(Stage::Preprocess)); 
					if (CheckTermination()) return;
					//mp_engine_core->GetFenceTrack().WaitForEndOf(?)
						

					// Main render
					SetState(State::Work);
					SetStage(Stage::MainRender);
					for (const auto& config : configs)
					{
						if (config.GetUpdateFlag())
						{
							CudaKernel::LaunchFirstPass
								<< <
								config.GetGrid(),
								config.GetThreadBlock(),
								config.GetSharedMemorySize(),
								mp_engine_core->GetRenderStream()
								>> >
								(mp_engine_core->GetGlobalKernel(mp_engine_core->GetIndexer().RenderIdx()),
									mp_engine_core->GetCudaWorld(),
									config.GetCameraId());
							CudaErrorCheck(cudaStreamSynchronize(mp_engine_core->GetRenderStream()));
							CudaErrorCheck(cudaGetLastError());
							m_time_table.AppendStage("main render");


							CudaKernel::SpacialReprojection
								<< <
								config.GetGrid(),
								config.GetThreadBlock(),
								0u,
								mp_engine_core->GetRenderStream()
								>> >
								(mp_engine_core->GetCudaWorld(), config.GetCameraId());
							CudaErrorCheck(cudaStreamSynchronize(mp_engine_core->GetRenderStream()));
							CudaErrorCheck(cudaGetLastError());
							m_time_table.AppendStage("reprojection");
						}
						else
						{
							CudaKernel::LaunchCumulativePass
								<< <
								config.GetGrid(),
								config.GetThreadBlock(),
								config.GetSharedMemorySize(),
								mp_engine_core->GetRenderStream()
								>> >
								(mp_engine_core->GetGlobalKernel(mp_engine_core->GetIndexer().RenderIdx()),
									mp_engine_core->GetCudaWorld(),
									config.GetCameraId());
							CudaErrorCheck(cudaStreamSynchronize(mp_engine_core->GetRenderStream()));
							CudaErrorCheck(cudaGetLastError());
							m_time_table.AppendStage("main render");
							m_time_table.AppendStage("reprojection");
						}
					}

					SetState(State::Wait);
					SetStage(Stage::None);
					m_fence_track.OpenGate(size_t(Stage::MainRender));
					mp_engine_core->GetFenceTrack().WaitForEndOf(size_t(CudaEngineCore::Stage::ResultTransfer));
					if (CheckTermination()) return;
					m_time_table.AppendStage("wait for postprocess");

					// Postprocess
					SetState(State::Work);
					SetStage(Stage::Postprocess);
					for (const auto& config : configs)
					{
						CudaKernel::ToneMap
							<< <
							config.GetGrid(),
							config.GetThreadBlock(),
							0u,
							mp_engine_core->GetRenderStream()
							>> >
							(mp_engine_core->GetGlobalKernel(mp_engine_core->GetIndexer().RenderIdx()),
								mp_engine_core->GetCudaWorld(), 
								config.GetCameraId());
						CudaErrorCheck(cudaStreamSynchronize(mp_engine_core->GetRenderStream()));
						CudaErrorCheck(cudaGetLastError());

						m_time_table.AppendStage("tone mapping");
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
				ReportException(
					Exception(__FILE__, __LINE__, L"Rendering function unknown fail."));
			}
		}
		bool CudaRenderer::CheckTermination()
		{
			return m_terminate_thread;
		}

		void CudaRenderer::ReportException(const Exception& e)
		{
			if (!m_exception)
			{
				m_exception.reset(new Exception(e));
			}
		}
		void CudaRenderer::ReportCudaException(const CudaException& e)
		{
			if (!m_cuda_exception)
			{
				m_cuda_exception.reset(new CudaException(e));
			}
		}
		void CudaRenderer::ResetExceptions()
		{
			m_exception = nullptr;
			m_cuda_exception = nullptr;
		}
		void CudaRenderer::ThrowIfException()
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
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


		// ~~~~~~~~ [STRUCT] CudaEngineCore ~~~~~~~~
		CudaEngineCore::CudaEngineCore()
			: mp_dCudaWorld(nullptr)
			, m_hpm_CudaWorld(sizeof(CudaWorld))
			, m_hpm_CudaKernel(std::max(sizeof(CudaGlobalKernel), sizeof(CudaConstantKernel)))
			, m_renderer(this)
			, m_fence_track(false)
		{
			cudaSetDevice(0);

			CreateStreams();
			CreateGlobalKernels();
			CreateCudaWorld();
		}
		CudaEngineCore::~CudaEngineCore()
		{
			DestroyCudaWorld();
			DestroyGlobalKernels();
			DestroyStreams();

			GetHardware().Reset();
		}


		void CudaEngineCore::RenderWorld(
			World& hWorld,
			const bool block,
			const bool sync)
		{
			std::lock_guard<std::mutex> lg(m_mtx);

			// check reported exceptions and throw if any
			m_renderer.ThrowIfException();
			m_renderer.LaunchThread();

			m_core_time_table.ResetTable();
			m_core_time_table.ResetTime();


			// [>] Async reconstruction
			SetState(State::Work);
			SetStage(Stage::AsyncReconstruction);

			// update host world
			m_update_flag = hWorld.GetStateRegister().RequiresUpdate();
			mp_hWorld = &hWorld;
			hWorld.Update();
			m_core_time_table.AppendStage("hWorld update");

			// create launch configurations
			m_configs[m_indexer.UpdateIdx()].Construct(m_hardware, hWorld, m_update_flag);
			m_core_time_table.AppendStage("configs construct");

			// reconstruct cuda kernels
			ReconstructKernels();
			m_core_time_table.AppendStage("kernels reconstruct");

			SetState(State::Wait);
			SetStage(Stage::None);
			m_fence_track.OpenGate(size_t(CudaEngineCore::Stage::AsyncReconstruction));
			m_renderer.GetFenceTrack().WaitForEndOf(size_t(CudaRenderer::Stage::MainRender));
			m_core_time_table.AppendStage("wait for main render");


			// [>] dCudaWorld async reconstruction
			SetState(State::Work);
			SetStage(Stage::WorldReconstruction);

			// reconstruct dCudaWorld
			//ReconstructCudaWorld();	// make this reconstruct all except cameras

			SetState(State::Wait);
			SetStage(Stage::None);
			m_fence_track.OpenGate(size_t(CudaEngineCore::Stage::WorldReconstruction));
			m_renderer.GetFenceTrack().WaitForEndOf(size_t(CudaRenderer::Stage::Postprocess));
			m_core_time_table.AppendStage("wait for postprocess");


			// [>] dCudaWorld sync reconstruction (CudaCamera reconstructions)
			SetState(State::Work);
			SetStage(Stage::CameraReconstruction);

			// reconstruct 
			ReconstructCudaWorld();	// TODO: make this rconstruct only cameras
			m_core_time_table.AppendStage("dCudaWorld reconstruct");

			// swap indices
			m_indexer.Swap();
			m_render_time_table = m_renderer.GetTimeTable();

			SetState(State::Wait);
			SetStage(Stage::None);
			m_fence_track.OpenGate(size_t(CudaEngineCore::Stage::CameraReconstruction));

			//m_fence_track.OpenGate(size_t(CudaEngineCore::Stage::ResultTransfer));
			//m_renderer.GetFenceTrack().WaitForEndOf(size_t(CudaRenderer::Stage::Postprocess));
			//m_time_table.AppendStage("sync wait");
			//m_renderer.GetFenceTrack().OpenGate(size_t(CudaRenderer::Stage::Postprocess));


			// [>] Transfer results to host side
			SetState(State::Work);
			SetStage(Stage::ResultTransfer);

			TransferResults();
			m_core_time_table.AppendStage("result tranfer");
			m_core_time_table.AppendFullCycle("full host cycle");

			SetState(State::None);
			SetStage(Stage::None);
			m_fence_track.OpenGate(size_t(CudaEngineCore::Stage::ResultTransfer));
		}
		void CudaEngineCore::TransferResults()
		{
			for (uint32_t i = 0u; i < mp_hWorld->Container<Camera>().GetCapacity(); ++i)
			{
				// check if hostCamera does exict
				const Handle<Camera>& hCamera = mp_hWorld->Container<Camera>()[i];
				if (!hCamera) continue;	// no camera at this address
				if (!hCamera->Enabled()) continue;	// camera is disabled


				// [>] Get CudaWorld from device
				CudaWorld* hCudaWorld = (CudaWorld*)m_hpm_CudaWorld.GetPointerToMemory();
				CudaErrorCheck(cudaMemcpyAsync(
					hCudaWorld, mp_dCudaWorld,
					sizeof(CudaWorld),
					cudaMemcpyKind::cudaMemcpyDeviceToHost, m_update_stream));
				CudaErrorCheck(cudaStreamSynchronize(m_update_stream));

				if (hCudaWorld->cameras.GetCount() == 0u) return;	// hCudaWorld has no cameras


				// [>] Get CudaCamera class from hCudaWorld
				CudaCamera* hCudaCamera = nullptr;
				if (CudaWorld::m_hpm.GetSize() < sizeof(*hCudaCamera))
					ThrowException(L"insufficient host pinned memory for CudaCamera");
				hCudaCamera = (CudaCamera*)CudaWorld::m_hpm.GetPointerToMemory();

				CudaErrorCheck(cudaMemcpyAsync(
					hCudaCamera, &hCudaWorld->cameras[i],
					sizeof(CudaCamera),
					cudaMemcpyKind::cudaMemcpyDeviceToHost, m_update_stream));
				CudaErrorCheck(cudaStreamSynchronize(m_update_stream));

				if (!hCudaCamera->Exist()) continue;


				// [>] Asynchronous copying
				hCamera->m_samples_count = hCudaCamera->GetPassesCount();

				static_assert(
					sizeof(*hCamera->GetImageBuffer().GetMapAddress()) ==
					sizeof(Color<unsigned char>),
					"sizeof(Graphics::Color) != sizeof(Color<unsigned char>)");

				// check cameras resolution
				if (hCamera->GetWidth() != hCudaCamera->GetWidth() ||
					hCamera->GetHeight() != hCudaCamera->GetHeight()) continue;

				uint32_t chunkSize =
					hCudaCamera->hostPinnedMemory.GetSize() /
					(sizeof(Color<unsigned char>));
				if (chunkSize < 16u) ThrowException(L"Not enough host pinned memory for async image copy");

				uint32_t nPixels = hCamera->GetWidth() * hCamera->GetHeight();
				for (uint32_t startIndex = 0; startIndex < nPixels; startIndex += chunkSize)
				{
					// find start index
					if (startIndex + chunkSize > nPixels) chunkSize = nPixels - startIndex;

					// find offset point
					Graphics::Point<uint32_t> offset_point(
						startIndex % hCamera->GetWidth(),
						startIndex / hCamera->GetWidth());

					// copy final image data from hCudaCamera to hCudaPixels on pinned memory
					Color<unsigned char>* hCudaPixels =
						(Color<unsigned char>*)CudaCamera::hostPinnedMemory.GetPointerToMemory();
					CudaErrorCheck(cudaMemcpyFromArrayAsync(
						hCudaPixels, hCudaCamera->FinalImageBuffer(m_indexer.UpdateIdx()).GetCudaArray(),
						offset_point.x * sizeof(*hCudaPixels), offset_point.y,
						chunkSize * sizeof(*hCudaPixels),
						cudaMemcpyKind::cudaMemcpyDeviceToHost, m_update_stream));
					CudaErrorCheck(cudaStreamSynchronize(m_update_stream));

					// copy final image data from hostCudaPixels on pinned memory to hostCamera
					hCamera->m_image_buffer.CopyFromMemory(
						hCudaPixels,
						chunkSize * sizeof(*hCudaPixels),
						offset_point.x, offset_point.y);


					// [>] Copy depth buffer
					float* hCudaDepthData =
						(float*)CudaCamera::hostPinnedMemory.GetPointerToMemory();
					CudaErrorCheck(cudaMemcpyFromArrayAsync(
						hCudaDepthData, hCudaCamera->FinalDepthBuffer(m_indexer.UpdateIdx()).GetCudaArray(),
						offset_point.x * sizeof(*hCudaDepthData), offset_point.y,
						chunkSize * sizeof(*hCudaDepthData),
						cudaMemcpyKind::cudaMemcpyDeviceToHost, m_update_stream));
					CudaErrorCheck(cudaStreamSynchronize(m_update_stream));

					// copy final image data from hostCudaPixels on pinned memory to hostCamera
					hCamera->m_depth_buffer.CopyFromMemory(
						hCudaDepthData,
						chunkSize * sizeof(*hCudaDepthData),
						offset_point.x, offset_point.y);
				}
			}
		}

		

		void CudaEngineCore::CreateStreams()
		{
			CudaErrorCheck(cudaStreamCreate(&m_update_stream));
			CudaErrorCheck(cudaStreamCreate(&m_render_stream));			
		}
		void CudaEngineCore::DestroyStreams()
		{
			CudaErrorCheck(cudaStreamDestroy(m_update_stream));
			CudaErrorCheck(cudaStreamDestroy(m_render_stream));
		}
		void CudaEngineCore::CreateGlobalKernels()
		{
			CudaGlobalKernel* hCudaGlobalKernel =
				(CudaGlobalKernel*)m_hpm_CudaKernel.GetPointerToMemory();
			for (uint32_t i = 0u; i < 2u; ++i)
			{
				new (hCudaGlobalKernel) CudaGlobalKernel();

				CudaErrorCheck(cudaMalloc(
					(void**)&mp_global_kernel[i], sizeof(CudaGlobalKernel)));
				CudaErrorCheck(cudaMemcpy(mp_global_kernel[i], hCudaGlobalKernel,
					sizeof(CudaGlobalKernel), cudaMemcpyKind::cudaMemcpyHostToDevice));
			}
		}
		void CudaEngineCore::DestroyGlobalKernels()
		{
			CudaGlobalKernel* hCudaKernelData =
				(CudaGlobalKernel*)m_hpm_CudaKernel.GetPointerToMemory();
			for (uint32_t i = 0u; i < 2u; ++i)
			{
				CudaErrorCheck(cudaMemcpy(
					hCudaKernelData, mp_global_kernel[i],
					sizeof(CudaGlobalKernel),
					cudaMemcpyKind::cudaMemcpyDeviceToHost));

				hCudaKernelData->~CudaGlobalKernel();

				CudaErrorCheck(cudaFree(mp_global_kernel[i]));
				mp_global_kernel[i] = nullptr;
			}
		}
		void CudaEngineCore::ReconstructKernels()
		{
			// [>] CudaGlobalKernel
			// get hpm memory
			CudaGlobalKernel* hCudaGlobalKernel =
				(CudaGlobalKernel*)m_hpm_CudaKernel.GetPointerToMemory();

			// copy dCudaKernelData to host
			CudaErrorCheck(cudaMemcpyAsync(
				hCudaGlobalKernel,
				mp_global_kernel[m_indexer.UpdateIdx()],
				sizeof(CudaGlobalKernel),
				cudaMemcpyKind::cudaMemcpyDeviceToHost,
				m_update_stream));
			CudaErrorCheck(cudaStreamSynchronize(m_update_stream));

			// reconstruct hCudaGlobalKernel
			hCudaGlobalKernel->Reconstruct(
				m_indexer.UpdateIdx(),
				m_update_stream);

			// copy hCudaGlobalKernel to device
			CudaErrorCheck(cudaMemcpyAsync(
				mp_global_kernel[m_indexer.UpdateIdx()],
				hCudaGlobalKernel,
				sizeof(CudaGlobalKernel),
				cudaMemcpyKind::cudaMemcpyHostToDevice,
				m_update_stream));
			CudaErrorCheck(cudaStreamSynchronize(m_update_stream));


			// [>] CudaConstantKernel
			// get hpm memory
			CudaConstantKernel* hCudaConstantKernel =
				(CudaConstantKernel*)m_hpm_CudaKernel.GetPointerToMemory();

			// reconstruct hCudaConstantKernel
			hCudaConstantKernel->Reconstruct();

			// copy hCudaConstantKernel to device __constant__ memory
			CudaKernel::CopyToConstantMemory(
				hCudaConstantKernel,
				m_indexer.UpdateIdx(), m_update_stream);
		}
		void CudaEngineCore::CreateCudaWorld()
		{
			CudaWorld* hCudaWorld = (CudaWorld*)m_hpm_CudaWorld.GetPointerToMemory();
			new (hCudaWorld) CudaWorld();
			CudaErrorCheck(cudaMalloc(&mp_dCudaWorld, sizeof(CudaWorld)));
			CudaErrorCheck(cudaMemcpy(
				mp_dCudaWorld, hCudaWorld,
				sizeof(CudaWorld),
				cudaMemcpyKind::cudaMemcpyHostToDevice));
		}
		void CudaEngineCore::DestroyCudaWorld()
		{
			if (mp_dCudaWorld)
			{
				CudaWorld* hCudaWorld = (CudaWorld*)m_hpm_CudaWorld.GetPointerToMemory();
				CudaErrorCheck(cudaMemcpy(
					hCudaWorld, mp_dCudaWorld,
					sizeof(CudaWorld),
					cudaMemcpyKind::cudaMemcpyDeviceToHost));

				hCudaWorld->~CudaWorld();

				CudaErrorCheck(cudaFree(mp_dCudaWorld));
				mp_dCudaWorld = nullptr;
			}
		}
		void CudaEngineCore::ReconstructCudaWorld()
		{
			if (!mp_hWorld->GetStateRegister().IsModified()) return;

			// copy CudaWorld to host
			CudaWorld* hCudaWorld = (CudaWorld*)m_hpm_CudaWorld.GetPointerToMemory();
			CudaErrorCheck(cudaMemcpyAsync(
				hCudaWorld, mp_dCudaWorld,
				sizeof(CudaWorld),
				cudaMemcpyKind::cudaMemcpyDeviceToHost, m_update_stream));
			CudaErrorCheck(cudaStreamSynchronize(m_update_stream));

			// reconstruct CudaWorld on host
			hCudaWorld->Reconstruct(*mp_hWorld, m_update_stream);

			// copy CudaWorld back to device
			CudaErrorCheck(cudaMemcpyAsync(
				mp_dCudaWorld, hCudaWorld,
				sizeof(CudaWorld),
				cudaMemcpyKind::cudaMemcpyHostToDevice, m_update_stream));
			CudaErrorCheck(cudaStreamSynchronize(m_update_stream));

			mp_hWorld->GetStateRegister().MakeUnmodified();
		}


		CudaHardware& CudaEngineCore::GetHardware()
		{
			return m_hardware;
		}
		CudaIndexer& CudaEngineCore::GetIndexer()
		{
			return m_indexer;
		}

		CudaRenderer& CudaEngineCore::GetRenderer()
		{
			return m_renderer;
		}
		LaunchConfigurations& CudaEngineCore::GetLaunchConfigs(const bool idx)
		{
			return m_configs[idx];
		}
		CudaGlobalKernel* CudaEngineCore::GetGlobalKernel(const bool idx)
		{
			return mp_global_kernel[idx];
		}
		CudaWorld* CudaEngineCore::GetCudaWorld()
		{
			return mp_dCudaWorld;
		}
		FenceTrack<5>& CudaEngineCore::GetFenceTrack()
		{
			return m_fence_track;
		}
		const TimeTable& CudaEngineCore::GetCoreTimeTable() const
		{
			return m_core_time_table;
		}
		const TimeTable& CudaEngineCore::GetRenderTimeTable() const
		{
			return m_render_time_table;
		}

		cudaStream_t& CudaEngineCore::GetUpdateStream()
		{
			return m_update_stream;
		}
		cudaStream_t& CudaEngineCore::GetRenderStream()
		{
			return m_render_stream;
		}

		const CudaEngineCore::State& CudaEngineCore::GetState()
		{
			return m_state;
		}
		const CudaEngineCore::Stage& CudaEngineCore::GetStage()
		{
			return m_stage;
		}
		void CudaEngineCore::SetState(const CudaEngineCore::State& state)
		{
			m_state = state;
		}
		void CudaEngineCore::SetStage(const CudaEngineCore::Stage& stage)
		{
			m_stage = stage;
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	}
}