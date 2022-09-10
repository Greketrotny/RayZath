#include "cuda_engine_core.cuh"

#include "cuda_kernel_data.cuh"

#include "point.h"

namespace RayZath::Cuda
{
	EngineCore::EngineCore()
		: mp_dCudaWorld(nullptr)
		, mp_hCudaWorld(nullptr)
		, m_hpm_CudaWorld(sizeof(World))
		, m_hpm_CudaKernel(std::max(sizeof(Kernel::GlobalKernel), sizeof(Kernel::ConstantKernel)))
		, m_renderer(this)
		, m_fence_track(false)
	{
		cudaSetDevice(0);

		CreateStreams();
		CreateGlobalKernels();
		CreateCudaWorld();
	}
	EngineCore::~EngineCore()
	{
		DestroyCudaWorld();
		DestroyGlobalKernels();
		DestroyStreams();

		//GetHardware().Reset();
	}

	void EngineCore::RenderWorld(
		RayZath::Engine::World& hWorld,
		const RayZath::Engine::RenderConfig& render_config,
		const bool block,
		const bool sync)
	{
		std::lock_guard<std::mutex> lg(m_mtx);

		static bool config_constructed = false;
		static bool kernels_constructed = false;
		static bool waited_for_main_render = false;
		static bool waited_for_postprocess = false;

		try
		{
			// check reported exceptions and throw if any
			m_renderer.ThrowIfException();
			m_renderer.LaunchThread();

			m_core_time_table.ResetTable();
			m_core_time_table.ResetTime();


			// [>] Async reconstruction
			SetState(State::Work);
			SetStage(Stage::AsyncReconstruction);

			if (!config_constructed || !waited_for_main_render)
			{
				// update host world
				m_update_flag = hWorld.GetStateRegister().RequiresUpdate();
				mp_hWorld = &hWorld;
				hWorld.Update();
				m_core_time_table.AppendStage("hWorld update");

				// create launch configurations
				m_configs[m_indexer.UpdateIdx()].Construct(m_hardware, hWorld, m_update_flag);
				m_core_time_table.AppendStage("configs construct");

				config_constructed = true;
			}

			// reconstruct cuda kernels
			if (!kernels_constructed)
			{
				m_render_config = render_config;
				ReconstructKernels();
				m_core_time_table.AppendStage("kernels reconstruct");
				m_fence_track.OpenGate(size_t(EngineCore::Stage::AsyncReconstruction));

				kernels_constructed = true;
			}

			if (!waited_for_main_render)
			{
				if (block ||
					m_renderer.GetFenceTrack().CheckGate(size_t(Renderer::Stage::MainRender)).State() ==
					Engine::ThreadGate::GateState::Opened)
				{
					SetState(State::Wait);
					m_renderer.GetFenceTrack().WaitForEndOfAndClose(size_t(Renderer::Stage::MainRender));
					m_core_time_table.AppendStage("wait for main render");

					// [>] dCudaWorld async reconstruction
					SetState(State::Work);
					SetStage(Stage::WorldReconstruction);

					if (mp_hWorld->GetStateRegister().IsModified())
					{
						// reconstruct resources and objects
						CopyCudaWorldDeviceToHost();
						mp_hCudaWorld->ReconstructResources(hWorld, m_update_stream);
						mp_hCudaWorld->ReconstructObjects(hWorld, m_render_config, m_update_stream);
					}
					m_core_time_table.AppendStage("objects reconstruct");
					m_fence_track.OpenGate(size_t(EngineCore::Stage::WorldReconstruction));
				}
				else return;

				waited_for_main_render = true;
			}

			if (!waited_for_postprocess)
			{
				if (block ||
					m_renderer.GetFenceTrack().CheckGate(size_t(Renderer::Stage::Postprocess)).State() ==
					Engine::ThreadGate::GateState::Opened)
				{
					// wait for postprocess to end
					SetState(State::Wait);
					m_renderer.GetFenceTrack().WaitForEndOfAndClose(size_t(Renderer::Stage::Postprocess));
					m_core_time_table.AppendStage("wait for postprocess");


					// [>] dCudaWorld sync reconstruction (Camera reconstructions)
					SetState(State::Work);
					SetStage(Stage::CameraReconstruction);

					if (mp_hWorld->GetStateRegister().IsModified())
					{
						// reconstruct cameras
						mp_hCudaWorld->ReconstructCameras(hWorld, m_update_stream);
						CopyCudaWorldHostToDevice();
						mp_hWorld->GetStateRegister().MakeUnmodified();
					}
					m_core_time_table.AppendStage("cameras reconstruct");

					m_fence_track.OpenGate(size_t(EngineCore::Stage::CameraReconstruction));
				}
				else return;

				waited_for_postprocess = true;
			}


			SetState(State::Wait);
			m_renderer.GetFenceTrack().WaitForEndOfAndClose(size_t(Renderer::Stage::Idle));


			// [>] Synchronize with renderer
			SetState(State::Work);
			SetStage(Stage::Synchronization);

			// swap indices
			m_indexer.Swap();
			m_render_time_table = m_renderer.GetTimeTable();

			m_fence_track.OpenGate(size_t(EngineCore::Stage::Synchronization));

			if (sync)
			{
				m_fence_track.OpenGate(size_t(EngineCore::Stage::ResultTransfer));
				SetState(State::Wait);
				m_renderer.GetFenceTrack().WaitForEndOf(size_t(Renderer::Stage::Postprocess));
				m_core_time_table.AppendStage("sync wait");
			}


			// [>] Transfer results to host side
			SetState(State::Work);
			SetStage(Stage::ResultTransfer);

			CopyRenderToHost();
			m_core_time_table.AppendStage("result tranfer");
			m_core_time_table.AppendFullCycle("full host cycle");

			SetState(State::None);
			SetStage(Stage::None);
			m_fence_track.OpenGate(size_t(EngineCore::Stage::ResultTransfer));
		}
		catch (...)
		{
			config_constructed =
				kernels_constructed =
				waited_for_main_render =
				waited_for_postprocess = false;
			throw;
		}

		config_constructed = 
			kernels_constructed = 
			waited_for_main_render = 
			waited_for_postprocess = false;
	}
	void EngineCore::CopyRenderToHost()
	{
		RZAssertCore(World::m_hpm.GetSize() >= sizeof(Camera), "insufficient host pinned memory for Camera");

		// [>] Get World from device
		World* hCudaWorld = (World*)m_hpm_CudaWorld.GetPointerToMemory();
		RZAssertCoreCUDA(cudaMemcpyAsync(
			hCudaWorld, mp_dCudaWorld,
			sizeof(World),
			cudaMemcpyKind::cudaMemcpyDeviceToHost, m_update_stream));
		RZAssertCoreCUDA(cudaStreamSynchronize(m_update_stream));

		if (hCudaWorld->cameras.GetCount() == 0u) return;	// hCudaWorld has no cameras


		const uint32_t count = std::min(
			hCudaWorld->cameras.GetCount(),
			mp_hWorld->Container<RayZath::Engine::World::ObjectType::Camera>().GetCount());
		for (uint32_t i = 0u; i < count; ++i)
		{
			// check if hostCamera is enabled
			const auto& hCamera = mp_hWorld->Container<RayZath::Engine::World::ObjectType::Camera>()[i];
			if (!hCamera) continue;	// no camera at this address
			if (!hCamera->Enabled()) continue;	// camera is disabled

			// [>] Get Camera class from hCudaWorld
			Camera* hCudaCamera = (Camera*)World::m_hpm.GetPointerToMemory();
			RZAssertCoreCUDA(cudaMemcpyAsync(
				hCudaCamera, &hCudaWorld->cameras[i],
				sizeof(Camera),
				cudaMemcpyKind::cudaMemcpyDeviceToHost, m_update_stream));
			RZAssertCoreCUDA(cudaStreamSynchronize(m_update_stream));


			// [>] Asynchronous copying
			hCamera->m_ray_count = hCudaCamera->GetResultRayCount();
			auto& hMeshes = mp_hWorld->Container<RayZath::Engine::World::ObjectType::Mesh>();
			auto& hMaterials = mp_hWorld->Container<RayZath::Engine::World::ObjectType::Material>();

			if (hCudaCamera->m_mesh_idx < hMeshes.GetCount())
			{
				auto& hRaycastedMesh = hMeshes[hCudaCamera->m_mesh_idx];
				hCamera->m_raycasted_mesh = hRaycastedMesh;

				if (hCudaCamera->m_mesh_material_idx < hRaycastedMesh->GetMaterialCapacity())
					hCamera->m_raycasted_material = hRaycastedMesh->GetMaterial(hCudaCamera->m_mesh_material_idx);
				else
					hCamera->m_raycasted_material.Release();
			}
			else
			{
				hCamera->m_raycasted_mesh.Release();
				hCamera->m_raycasted_material.Release();
			}

			static_assert(
				sizeof(*hCamera->GetImageBuffer().GetMapAddress()) ==
				sizeof(Color<unsigned char>),
				"sizeof(Graphics::Color) != sizeof(Color<unsigned char>)");

			// check cameras resolution
			if (hCamera->GetWidth() != hCudaCamera->GetWidth() ||
				hCamera->GetHeight() != hCudaCamera->GetHeight()) continue;

			uint32_t chunkSize = uint32_t(
				hCudaCamera->hostPinnedMemory.GetSize() /
				(sizeof(Color<unsigned char>)));
			RZAssertCore(chunkSize != 0u, "Not enough host pinned memory for async image copy");

			const uint32_t nPixels = hCamera->GetWidth() * hCamera->GetHeight();
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
					(Color<unsigned char>*)Camera::hostPinnedMemory.GetPointerToMemory();
				RZAssertCoreCUDA(cudaMemcpyFromArrayAsync(
					hCudaPixels, hCudaCamera->FinalImageBuffer().GetCudaArray(),
					offset_point.x * sizeof(*hCudaPixels), offset_point.y,
					chunkSize * sizeof(*hCudaPixels),
					cudaMemcpyKind::cudaMemcpyDeviceToHost, m_update_stream));
				RZAssertCoreCUDA(cudaStreamSynchronize(m_update_stream));

				// copy final image data from hostCudaPixels on pinned memory to hostCamera
				hCamera->m_image_buffer.CopyFromMemory(
					hCudaPixels,
					chunkSize * sizeof(*hCudaPixels),
					offset_point.x, offset_point.y);


				// [>] Copy depth buffer
				float* hCudaDepthData =
					(float*)Camera::hostPinnedMemory.GetPointerToMemory();
				RZAssertCoreCUDA(cudaMemcpyFromArrayAsync(
					hCudaDepthData, hCudaCamera->FinalDepthBuffer().GetCudaArray(),
					offset_point.x * sizeof(*hCudaDepthData), offset_point.y,
					chunkSize * sizeof(*hCudaDepthData),
					cudaMemcpyKind::cudaMemcpyDeviceToHost, m_update_stream));
				RZAssertCoreCUDA(cudaStreamSynchronize(m_update_stream));

				// copy final image data from hostCudaPixels on pinned memory to hostCamera
				hCamera->m_depth_buffer.CopyFromMemory(
					hCudaDepthData,
					chunkSize * sizeof(*hCudaDepthData),
					offset_point.x, offset_point.y);
			}
		}
	}


	void EngineCore::CreateStreams()
	{
		RZAssertCoreCUDA(cudaStreamCreate(&m_update_stream));
		RZAssertCoreCUDA(cudaStreamCreate(&m_render_stream));
	}
	void EngineCore::DestroyStreams()
	{
		RZAssertCoreCUDA(cudaStreamDestroy(m_update_stream));
		RZAssertCoreCUDA(cudaStreamDestroy(m_render_stream));
	}
	void EngineCore::CreateGlobalKernels()
	{
		Kernel::GlobalKernel* hCudaGlobalKernel =
			(Kernel::GlobalKernel*)m_hpm_CudaKernel.GetPointerToMemory();
		for (uint32_t i = 0u; i < 2u; ++i)
		{
			new (hCudaGlobalKernel) Kernel::GlobalKernel();

			RZAssertCoreCUDA(cudaMalloc(
				(void**)&mp_global_kernel[i], sizeof(Kernel::GlobalKernel)));
			RZAssertCoreCUDA(cudaMemcpy(mp_global_kernel[i], hCudaGlobalKernel,
				sizeof(Kernel::GlobalKernel), cudaMemcpyKind::cudaMemcpyHostToDevice));
		}
	}
	void EngineCore::DestroyGlobalKernels()
	{
		Kernel::GlobalKernel* hCudaKernelData =
			(Kernel::GlobalKernel*)m_hpm_CudaKernel.GetPointerToMemory();
		for (uint32_t i = 0u; i < 2u; ++i)
		{
			RZAssertCoreCUDA(cudaMemcpy(
				hCudaKernelData, mp_global_kernel[i],
				sizeof(Kernel::GlobalKernel),
				cudaMemcpyKind::cudaMemcpyDeviceToHost));

			hCudaKernelData->~GlobalKernel();

			RZAssertCoreCUDA(cudaFree(mp_global_kernel[i]));
			mp_global_kernel[i] = nullptr;
		}
	}
	void EngineCore::ReconstructKernels()
	{
		// [>] GlobalKernel
		// get hpm memory
		Kernel::GlobalKernel* hCudaGlobalKernel =
			(Kernel::GlobalKernel*)m_hpm_CudaKernel.GetPointerToMemory();

		// copy dCudaKernelData to host
		RZAssertCoreCUDA(cudaMemcpyAsync(
			hCudaGlobalKernel,
			mp_global_kernel[m_indexer.UpdateIdx()],
			sizeof(Kernel::GlobalKernel),
			cudaMemcpyKind::cudaMemcpyDeviceToHost,
			m_update_stream));
		RZAssertCoreCUDA(cudaStreamSynchronize(m_update_stream));

		// reconstruct hCudaGlobalKernel
		hCudaGlobalKernel->Reconstruct(
			m_indexer.UpdateIdx(),
			m_update_stream);

		// copy hCudaGlobalKernel to device
		RZAssertCoreCUDA(cudaMemcpyAsync(
			mp_global_kernel[m_indexer.UpdateIdx()],
			hCudaGlobalKernel,
			sizeof(Kernel::GlobalKernel),
			cudaMemcpyKind::cudaMemcpyHostToDevice,
			m_update_stream));
		RZAssertCoreCUDA(cudaStreamSynchronize(m_update_stream));


		// [>] ConstantKernel
		// get hpm memory
		Kernel::ConstantKernel* hCudaConstantKernel =
			(Kernel::ConstantKernel*)m_hpm_CudaKernel.GetPointerToMemory();

		// reconstruct hCudaConstantKernel
		hCudaConstantKernel->Reconstruct(
			m_render_config);

		// copy hCudaConstantKernel to device __constant__ memory
		Kernel::CopyConstantKernel(
			hCudaConstantKernel,
			m_indexer.UpdateIdx(), m_update_stream);
	}
	void EngineCore::CreateCudaWorld()
	{
		mp_hCudaWorld = (World*)m_hpm_CudaWorld.GetPointerToMemory();
		new (mp_hCudaWorld) World();
		RZAssertCoreCUDA(cudaMalloc(&mp_dCudaWorld, sizeof(World)));
		CopyCudaWorldHostToDevice();
	}
	void EngineCore::DestroyCudaWorld()
	{
		if (mp_dCudaWorld)
		{
			CopyCudaWorldDeviceToHost();
			mp_hCudaWorld->~World();
			RZAssertCoreCUDA(cudaFree(mp_dCudaWorld));
			mp_dCudaWorld = nullptr;
		}
	}
	void EngineCore::CopyCudaWorldDeviceToHost()
	{
		mp_hCudaWorld = (World*)m_hpm_CudaWorld.GetPointerToMemory();
		RZAssertCoreCUDA(cudaMemcpyAsync(
			mp_hCudaWorld, mp_dCudaWorld,
			sizeof(World),
			cudaMemcpyKind::cudaMemcpyDeviceToHost, m_update_stream));
		RZAssertCoreCUDA(cudaStreamSynchronize(m_update_stream));
	}
	void EngineCore::CopyCudaWorldHostToDevice()
	{
		if (mp_dCudaWorld && mp_hCudaWorld)
		{
			RZAssertCoreCUDA(cudaMemcpyAsync(
				mp_dCudaWorld, mp_hCudaWorld,
				sizeof(World),
				cudaMemcpyKind::cudaMemcpyHostToDevice, m_update_stream));
			RZAssertCoreCUDA(cudaStreamSynchronize(m_update_stream));
		}
	}


	Hardware& EngineCore::GetHardware()
	{
		return m_hardware;
	}
	Indexer& EngineCore::GetIndexer()
	{
		return m_indexer;
	}

	Renderer& EngineCore::GetRenderer()
	{
		return m_renderer;
	}
	LaunchConfigurations& EngineCore::GetLaunchConfigs(const bool idx)
	{
		return m_configs[idx];
	}
	Kernel::GlobalKernel* EngineCore::GetGlobalKernel(const bool idx)
	{
		return mp_global_kernel[idx];
	}
	const RayZath::Engine::RenderConfig& EngineCore::GetRenderConfig() const
	{
		return m_render_config;
	}
	World* EngineCore::GetCudaWorld()
	{
		return mp_dCudaWorld;
	}
	EngineCore::FenceTrack_t& EngineCore::GetFenceTrack()
	{
		return m_fence_track;
	}
	const TimeTable& EngineCore::GetCoreTimeTable() const
	{
		return m_core_time_table;
	}
	const TimeTable& EngineCore::GetRenderTimeTable() const
	{
		return m_render_time_table;
	}

	cudaStream_t& EngineCore::GetUpdateStream()
	{
		return m_update_stream;
	}
	cudaStream_t& EngineCore::GetRenderStream()
	{
		return m_render_stream;
	}

	const EngineCore::State& EngineCore::GetState()
	{
		return m_state;
	}
	const EngineCore::Stage& EngineCore::GetStage()
	{
		return m_stage;
	}
	void EngineCore::SetState(const EngineCore::State& state)
	{
		m_state = state;
	}
	void EngineCore::SetStage(const EngineCore::Stage& stage)
	{
		m_stage = stage;
	}
}