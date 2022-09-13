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

		createStreams();
		createGlobalKernels();
		createCudaWorld();
	}
	EngineCore::~EngineCore()
	{
		destroyCudaWorld();
		destroyGlobalKernels();
		destroyStreams();

		//GetHardware().Reset();
	}

	void EngineCore::renderWorld(
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
			m_renderer.throwIfException();
			m_renderer.launchThread();

			m_core_time_table.resetTable();
			m_core_time_table.resetTime();


			// [>] Async reconstruction
			setState(State::Work);
			setStage(Stage::AsyncReconstruction);

			if (!config_constructed || !waited_for_main_render)
			{
				// update host world
				m_update_flag = hWorld.stateRegister().RequiresUpdate();
				mp_hWorld = &hWorld;
				hWorld.update();
				m_core_time_table.appendStage("hWorld update");

				// create launch configurations
				m_configs[m_indexer.updateIdx()].construct(m_hardware, hWorld, m_update_flag);
				m_core_time_table.appendStage("configs construct");

				config_constructed = true;
			}

			// reconstruct cuda kernels
			if (!kernels_constructed)
			{
				m_render_config = render_config;
				reconstructKernels();
				m_core_time_table.appendStage("kernels reconstruct");
				m_fence_track.openGate(size_t(EngineCore::Stage::AsyncReconstruction));

				kernels_constructed = true;
			}

			if (!waited_for_main_render)
			{
				if (block ||
					m_renderer.fenceTrack().checkGate(size_t(Renderer::Stage::MainRender)).state() ==
					Engine::ThreadGate::GateState::Opened)
				{
					setState(State::wait);
					m_renderer.fenceTrack().waitForEndOfAndClose(size_t(Renderer::Stage::MainRender));
					m_core_time_table.appendStage("wait for main render");

					// [>] dCudaWorld async reconstruction
					setState(State::Work);
					setStage(Stage::WorldReconstruction);

					if (mp_hWorld->stateRegister().IsModified())
					{
						// reconstruct resources and objects
						copyCudaWorldDeviceToHost();
						mp_hCudaWorld->reconstructResources(hWorld, m_update_stream);
						mp_hCudaWorld->reconstructObjects(hWorld, m_render_config, m_update_stream);
					}
					m_core_time_table.appendStage("objects reconstruct");
					m_fence_track.openGate(size_t(EngineCore::Stage::WorldReconstruction));
				}
				else return;

				waited_for_main_render = true;
			}

			if (!waited_for_postprocess)
			{
				if (block ||
					m_renderer.fenceTrack().checkGate(size_t(Renderer::Stage::Postprocess)).state() ==
					Engine::ThreadGate::GateState::Opened)
				{
					// wait for postprocess to end
					setState(State::wait);
					m_renderer.fenceTrack().waitForEndOfAndClose(size_t(Renderer::Stage::Postprocess));
					m_core_time_table.appendStage("wait for postprocess");


					// [>] dCudaWorld sync reconstruction (Camera reconstructions)
					setState(State::Work);
					setStage(Stage::CameraReconstruction);

					if (mp_hWorld->stateRegister().IsModified())
					{
						// reconstruct cameras
						mp_hCudaWorld->reconstructCameras(hWorld, m_update_stream);
						copyCudaWorldHostToDevice();
						mp_hWorld->stateRegister().MakeUnmodified();
					}
					m_core_time_table.appendStage("cameras reconstruct");

					m_fence_track.openGate(size_t(EngineCore::Stage::CameraReconstruction));
				}
				else return;

				waited_for_postprocess = true;
			}


			setState(State::wait);
			m_renderer.fenceTrack().waitForEndOfAndClose(size_t(Renderer::Stage::Idle));


			// [>] Synchronize with renderer
			setState(State::Work);
			setStage(Stage::Synchronization);

			// swap indices
			m_indexer.swap();
			m_render_time_table = m_renderer.timeTable();

			m_fence_track.openGate(size_t(EngineCore::Stage::Synchronization));

			if (sync)
			{
				m_fence_track.openGate(size_t(EngineCore::Stage::ResultTransfer));
				setState(State::wait);
				m_renderer.fenceTrack().waitForEndOf(size_t(Renderer::Stage::Postprocess));
				m_core_time_table.appendStage("sync wait");
			}


			// [>] Transfer results to host side
			setState(State::Work);
			setStage(Stage::ResultTransfer);

			CopyRenderToHost();
			m_core_time_table.appendStage("result tranfer");
			m_core_time_table.appendFullCycle("full host cycle");

			setState(State::None);
			setStage(Stage::None);
			m_fence_track.openGate(size_t(EngineCore::Stage::ResultTransfer));
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
		RZAssertCore(World::m_hpm.size() >= sizeof(Camera), "insufficient host pinned memory for Camera");

		// [>] Get World from device
		World* hCudaWorld = (World*)m_hpm_CudaWorld.GetPointerToMemory();
		RZAssertCoreCUDA(cudaMemcpyAsync(
			hCudaWorld, mp_dCudaWorld,
			sizeof(World),
			cudaMemcpyKind::cudaMemcpyDeviceToHost, m_update_stream));
		RZAssertCoreCUDA(cudaStreamSynchronize(m_update_stream));

		if (hCudaWorld->cameras.count() == 0u) return;	// hCudaWorld has no cameras


		const uint32_t count = std::min(
			hCudaWorld->cameras.count(),
			mp_hWorld->container<RayZath::Engine::World::ObjectType::Camera>().count());
		for (uint32_t i = 0u; i < count; ++i)
		{
			// check if hostCamera is enabled
			const auto& hCamera = mp_hWorld->container<RayZath::Engine::World::ObjectType::Camera>()[i];
			if (!hCamera) continue;	// no camera at this address
			if (!hCamera->enabled()) continue;	// camera is disabled

			// [>] Get Camera class from hCudaWorld
			Camera* hCudaCamera = (Camera*)World::m_hpm.GetPointerToMemory();
			RZAssertCoreCUDA(cudaMemcpyAsync(
				hCudaCamera, &hCudaWorld->cameras[i],
				sizeof(Camera),
				cudaMemcpyKind::cudaMemcpyDeviceToHost, m_update_stream));
			RZAssertCoreCUDA(cudaStreamSynchronize(m_update_stream));


			// [>] Asynchronous copying
			hCamera->m_ray_count = hCudaCamera->getResultRayCount();
			auto& hMeshes = mp_hWorld->container<RayZath::Engine::World::ObjectType::Mesh>();

			if (hCudaCamera->m_mesh_idx < hMeshes.count())
			{
				auto& hRaycastedMesh = hMeshes[hCudaCamera->m_mesh_idx];
				hCamera->m_raycasted_mesh = hRaycastedMesh;

				if (hCudaCamera->m_mesh_material_idx < hRaycastedMesh->materialCapacity())
					hCamera->m_raycasted_material = hRaycastedMesh->material(hCudaCamera->m_mesh_material_idx);
				else
					hCamera->m_raycasted_material.release();
			}
			else
			{
				hCamera->m_raycasted_mesh.release();
				hCamera->m_raycasted_material.release();
			}

			static_assert(
				sizeof(*hCamera->imageBuffer().GetMapAddress()) ==
				sizeof(Color<unsigned char>),
				"sizeof(Graphics::Color) != sizeof(Color<unsigned char>)");

			// check cameras resolution
			if (hCamera->width() != hCudaCamera->width() ||
				hCamera->height() != hCudaCamera->height()) continue;

			uint32_t chunkSize = uint32_t(
				hCudaCamera->hostPinnedMemory.size() /
				(sizeof(Color<unsigned char>)));
			RZAssertCore(chunkSize != 0u, "Not enough host pinned memory for async image copy");

			const uint32_t nPixels = hCamera->width() * hCamera->height();
			for (uint32_t startIndex = 0; startIndex < nPixels; startIndex += chunkSize)
			{
				// find start index
				if (startIndex + chunkSize > nPixels) chunkSize = nPixels - startIndex;

				// find offset point
				Graphics::Point<uint32_t> offset_point(
					startIndex % hCamera->width(),
					startIndex / hCamera->width());

				// copy final image data from hCudaCamera to hCudaPixels on pinned memory
				Color<unsigned char>* hCudaPixels =
					(Color<unsigned char>*)Camera::hostPinnedMemory.GetPointerToMemory();
				RZAssertCoreCUDA(cudaMemcpyFromArrayAsync(
					hCudaPixels, hCudaCamera->finalImageBuffer().getCudaArray(),
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
					hCudaDepthData, hCudaCamera->finalDepthBuffer().getCudaArray(),
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


	void EngineCore::createStreams()
	{
		RZAssertCoreCUDA(cudaStreamCreate(&m_update_stream));
		RZAssertCoreCUDA(cudaStreamCreate(&m_render_stream));
	}
	void EngineCore::destroyStreams()
	{
		RZAssertCoreCUDA(cudaStreamDestroy(m_update_stream));
		RZAssertCoreCUDA(cudaStreamDestroy(m_render_stream));
	}
	void EngineCore::createGlobalKernels()
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
	void EngineCore::destroyGlobalKernels()
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
	void EngineCore::reconstructKernels()
	{
		// [>] GlobalKernel
		// get hpm memory
		Kernel::GlobalKernel* hCudaGlobalKernel =
			(Kernel::GlobalKernel*)m_hpm_CudaKernel.GetPointerToMemory();

		// copy dCudaKernelData to host
		RZAssertCoreCUDA(cudaMemcpyAsync(
			hCudaGlobalKernel,
			mp_global_kernel[m_indexer.updateIdx()],
			sizeof(Kernel::GlobalKernel),
			cudaMemcpyKind::cudaMemcpyDeviceToHost,
			m_update_stream));
		RZAssertCoreCUDA(cudaStreamSynchronize(m_update_stream));

		// reconstruct hCudaGlobalKernel
		hCudaGlobalKernel->reconstruct(
			m_indexer.updateIdx(),
			m_update_stream);

		// copy hCudaGlobalKernel to device
		RZAssertCoreCUDA(cudaMemcpyAsync(
			mp_global_kernel[m_indexer.updateIdx()],
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
		hCudaConstantKernel->reconstruct(
			m_render_config);

		// copy hCudaConstantKernel to device __constant__ memory
		Kernel::copyConstantKernel(
			hCudaConstantKernel,
			m_indexer.updateIdx(), m_update_stream);
	}
	void EngineCore::createCudaWorld()
	{
		mp_hCudaWorld = (World*)m_hpm_CudaWorld.GetPointerToMemory();
		new (mp_hCudaWorld) World();
		RZAssertCoreCUDA(cudaMalloc(&mp_dCudaWorld, sizeof(World)));
		copyCudaWorldHostToDevice();
	}
	void EngineCore::destroyCudaWorld()
	{
		if (mp_dCudaWorld)
		{
			copyCudaWorldDeviceToHost();
			mp_hCudaWorld->~World();
			RZAssertCoreCUDA(cudaFree(mp_dCudaWorld));
			mp_dCudaWorld = nullptr;
		}
	}
	void EngineCore::copyCudaWorldDeviceToHost()
	{
		mp_hCudaWorld = (World*)m_hpm_CudaWorld.GetPointerToMemory();
		RZAssertCoreCUDA(cudaMemcpyAsync(
			mp_hCudaWorld, mp_dCudaWorld,
			sizeof(World),
			cudaMemcpyKind::cudaMemcpyDeviceToHost, m_update_stream));
		RZAssertCoreCUDA(cudaStreamSynchronize(m_update_stream));
	}
	void EngineCore::copyCudaWorldHostToDevice()
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


	Hardware& EngineCore::hardware()
	{
		return m_hardware;
	}
	Indexer& EngineCore::indexer()
	{
		return m_indexer;
	}

	Renderer& EngineCore::renderer()
	{
		return m_renderer;
	}
	LaunchConfigurations& EngineCore::launchConfigs(const bool idx)
	{
		return m_configs[idx];
	}
	Kernel::GlobalKernel* EngineCore::globalKernel(const bool idx)
	{
		return mp_global_kernel[idx];
	}
	const RayZath::Engine::RenderConfig& EngineCore::renderConfig() const
	{
		return m_render_config;
	}
	World* EngineCore::cudaWorld()
	{
		return mp_dCudaWorld;
	}
	EngineCore::FenceTrack_t& EngineCore::fenceTrack()
	{
		return m_fence_track;
	}
	const TimeTable& EngineCore::coreTimeTable() const
	{
		return m_core_time_table;
	}
	const TimeTable& EngineCore::renderTimeTable() const
	{
		return m_render_time_table;
	}

	cudaStream_t& EngineCore::updateStream()
	{
		return m_update_stream;
	}
	cudaStream_t& EngineCore::renderStream()
	{
		return m_render_stream;
	}

	const EngineCore::State& EngineCore::state()
	{
		return m_state;
	}
	const EngineCore::Stage& EngineCore::stage()
	{
		return m_stage;
	}
	void EngineCore::setState(const EngineCore::State& state)
	{
		m_state = state;
	}
	void EngineCore::setStage(const EngineCore::Stage& stage)
	{
		m_stage = stage;
	}
}