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

		// check reported exceptions and throw if any
		m_renderer.throwIfException();
		m_renderer.launchThread();

		m_core_time_table.update("no render cycle");

		// [>] Async reconstruction
		// update host world
		const bool world_update = hWorld.stateRegister().IsModified();
		auto& hCameras = hWorld.container<Engine::World::ObjectType::Camera>();
		const bool cameras_update = hCameras.stateRegister().IsModified();
		const bool any_update_flag = world_update || cameras_update;
		if (world_update)
		{
			for (uint32_t i = 0; i < hCameras.count(); i++)
				hCameras[i]->stateRegister().RequestUpdate();
		}

		mp_hWorld = &hWorld;
		hWorld.update();
		hCameras.update();
		m_core_time_table.update("update world");

		// create launch configurations
		m_configs[m_indexer.updateIdx()].construct(m_hardware, hWorld, world_update);
		m_core_time_table.update("construct configs");

		// reconstruct cuda kernels
		m_render_config = render_config;
		reconstructKernels();
		m_core_time_table.update("reconstruct kernels");
		m_fence_track.openGate(size_t(EngineCore::Stage::AsyncReconstruction));

		// [>] dCudaWorld async reconstruction
		// wait for main render to finish
		m_core_time_table.setWaitTime(
			"reconstruct objects",
			m_renderer.fenceTrack().waitFor(size_t(Renderer::Stage::MainRender)));
		if (any_update_flag)
		{
			copyCudaWorldDeviceToHost();
		}
		if (mp_hWorld->stateRegister().IsModified())
		{
			// reconstruct resources and objects			
			mp_hCudaWorld->reconstructResources(hWorld, m_update_stream);
			mp_hCudaWorld->reconstructObjects(hWorld, m_render_config, m_update_stream);
		}
		m_fence_track.openGate(size_t(EngineCore::Stage::WorldReconstruction));
		m_core_time_table.update("reconstruct objects");


		// [>] dCudaWorld sync reconstruction (Camera reconstructions)
		// wait for postprocess to end
		m_core_time_table.setWaitTime(
			"reconstruct cameras",
			m_renderer.fenceTrack().waitFor(size_t(Renderer::Stage::Postprocess)));

		// reconstruct cameras
		mp_hCudaWorld->reconstructCameras(hWorld, m_update_stream);
		mp_hWorld->stateRegister().MakeUnmodified();

		if (any_update_flag)
		{
			copyCudaWorldHostToDevice();
		}
		m_core_time_table.update("reconstruct cameras");


		// [>] Synchronize with renderer
		// swap indices
		m_indexer.swap();
		m_render_time_table = m_renderer.timeTable();
		m_fence_track.openGate(size_t(EngineCore::Stage::Synchronization));

		if (sync)
		{
			m_fence_track.openGate(size_t(EngineCore::Stage::ResultTransfer));
			m_renderer.fenceTrack().waitForKeepOpen(size_t(Renderer::Stage::Postprocess));
			m_core_time_table.update("sync wait");
		}


		// [>] Transfer results to host side
		CopyRenderToHost();
		m_fence_track.openGate(size_t(EngineCore::Stage::ResultTransfer));
		m_core_time_table.update("result tranfer");
		m_core_time_table.updateCycle("full cycle");
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
			auto& hInstances = mp_hWorld->container<RayZath::Engine::World::ObjectType::Instance>();

			if (hCudaCamera->m_instance_idx < hInstances.count())
			{
				auto& hRaycastedInstance = hInstances[hCudaCamera->m_instance_idx];
				hCamera->m_raycasted_instance = hRaycastedInstance;

				if (hCudaCamera->m_instance_material_idx < hRaycastedInstance->materialCapacity())
					hCamera->m_raycasted_material = hRaycastedInstance->material(hCudaCamera->m_instance_material_idx);
				else
					hCamera->m_raycasted_material.release();
			}
			else
			{
				hCamera->m_raycasted_instance.release();
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
	const RayZath::Engine::TimeTable& EngineCore::coreTimeTable() const
	{
		return m_core_time_table;
	}
	const RayZath::Engine::TimeTable& EngineCore::renderTimeTable() const
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
}
