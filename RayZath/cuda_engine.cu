#include "cuda_engine.cuh"

namespace RayZath
{
	CudaEngine::CudaEngine()
		: mp_dCudaWorld(nullptr)
		, m_hpm_CudaWorld(sizeof(CudaWorld))
		, m_update_flag(true)
	{
		cudaSetDevice(0);

		// create streams
		CudaErrorCheck(cudaStreamCreate(&m_mirror_stream));
		CudaErrorCheck(cudaStreamCreate(&m_render_stream));

		// create empty dCudaWorld
		CudaWorld* hCudaWorld = (CudaWorld*)m_hpm_CudaWorld.GetPointerToMemory();
		new (hCudaWorld) CudaWorld();
		CudaErrorCheck(cudaMalloc(&mp_dCudaWorld, sizeof(CudaWorld)));
		CudaErrorCheck(cudaMemcpy(
			mp_dCudaWorld, hCudaWorld,
			sizeof(CudaWorld),
			cudaMemcpyKind::cudaMemcpyHostToDevice));
	}
	CudaEngine::~CudaEngine()
	{
		// destroy mp_dCudaWorld
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

		// destroy streams
		CudaErrorCheck(cudaStreamDestroy(m_mirror_stream));
		CudaErrorCheck(cudaStreamDestroy(m_render_stream));
	}

	void CudaEngine::RenderWorld(World& hWorld)
	{
		mainDebugInfo.Clear();

		m_update_flag = hWorld.RequiresUpdate();
		Timer function_timer, step_timer;
		std::wstring timing_string = L"Host side:\n";

		// [>] Create Launch configurations
		step_timer.Start();
		CreateLaunchConfigurations(hWorld);
		AppendTimeToString(timing_string, L"create launch configs: ", step_timer.GetElapsedTime());


		// [>] Reconstruct dCudaWorld
		step_timer.Start();
		if (hWorld.RequiresUpdate())
		{
			ReconstructCudaWorld(mp_dCudaWorld, hWorld, &m_mirror_stream);
			hWorld.Updated();
		}
		AppendTimeToString(timing_string, L"reconstruct CudaWorld: ", step_timer.GetElapsedTime());


		LaunchFunction();


		// [>] Transfer results to host
		step_timer.Start();
		TransferResultsToHost(mp_dCudaWorld, hWorld, &m_mirror_stream);
		AppendTimeToString(timing_string, L"copy final render to host: ", step_timer.GetElapsedTime());


		// [>] Sum up timings and add debug string
		AppendTimeToString(timing_string, L"render function full time: ", function_timer.GetElapsedTime());
		mainDebugInfo.AddDebugString(timing_string);
	}
	void CudaEngine::CreateLaunchConfigurations(const World& world)
	{
		m_launch_configs[m_update_ix].clear();
		for (size_t i = 0; i < world.GetCameras().GetCapacity(); ++i)
		{
			Camera* camera = world.GetCameras()[i];
			if (camera == nullptr) continue;	// no camera at the index
			if (!camera->Enabled()) continue;	// camera is disabled

			m_launch_configs[m_update_ix].push_back(
				LaunchConfiguration(
					m_hardware, *camera, m_update_flag));
		}
	}
	void CudaEngine::ReconstructCudaWorld(
		CudaWorld* dCudaWorld,
		World& hWorld,
		cudaStream_t* mirror_stream)
	{
		// copy CudaWorld to host
		CudaWorld* hCudaWorld = (CudaWorld*)m_hpm_CudaWorld.GetPointerToMemory();
		CudaErrorCheck(cudaMemcpyAsync(
			hCudaWorld, dCudaWorld,
			sizeof(CudaWorld),
			cudaMemcpyKind::cudaMemcpyDeviceToHost, *mirror_stream));
		CudaErrorCheck(cudaStreamSynchronize(*mirror_stream));

		// reconstruct CudaWorld on host
		hCudaWorld->Reconstruct(hWorld, mirror_stream);

		// copy CudaWorld back to device
		CudaErrorCheck(cudaMemcpyAsync(
			dCudaWorld, hCudaWorld,
			sizeof(CudaWorld),
			cudaMemcpyKind::cudaMemcpyHostToDevice, *mirror_stream));
		CudaErrorCheck(cudaStreamSynchronize(*mirror_stream));

		hWorld.Updated();
	}
	void CudaEngine::TransferResultsToHost(
		CudaWorld* dCudaWorld,
		World& hWorld,
		cudaStream_t* mirror_stream)
	{
		for (size_t i = 0; i < hWorld.GetCameras().GetCapacity(); ++i)
		{
			// check if hostCamera does exict
			Camera* hCamera = hWorld.GetCameras()[i];
			if (hCamera == nullptr) continue;	// no camera at this address
			if (!hCamera->Enabled()) continue;	// camera is disabled


			// [>] Get CudaWorld from device
			CudaWorld* hCudaWorld = (CudaWorld*)m_hpm_CudaWorld.GetPointerToMemory();
			CudaErrorCheck(cudaMemcpyAsync(
				hCudaWorld, dCudaWorld, 
				sizeof(CudaWorld), 
				cudaMemcpyKind::cudaMemcpyDeviceToHost, *mirror_stream));
			CudaErrorCheck(cudaStreamSynchronize(*mirror_stream));

			if (hCudaWorld->cameras.GetCount() == 0) return;	// hCudaWorld has no cameras


			// [>] Get CudaCamera class from hCudaWorld
			CudaCamera* hCudaCamera = nullptr;
			if (CudaWorld::m_hpm.GetSize() < sizeof(*hCudaCamera))
				ThrowException(L"insufficient host pinned memory for CudaCamera");
			hCudaCamera = (CudaCamera*)CudaWorld::m_hpm.GetPointerToMemory();

			CudaErrorCheck(cudaMemcpyAsync(
				hCudaCamera, &hCudaWorld->cameras[i], 
				sizeof(CudaCamera), 
				cudaMemcpyKind::cudaMemcpyDeviceToHost, *mirror_stream));
			CudaErrorCheck(cudaStreamSynchronize(*mirror_stream));

			if (!hCudaCamera->Exist()) continue;


			// [>] Asynchronous copying
			hCamera->m_samples_count = hCudaCamera->samples_count;

			static_assert(
				sizeof(*hCamera->GetBitmap().GetMapAddress()) == 
				sizeof(*hCudaCamera->final_image[m_update_ix]), 
				"sizeof(Graphics::Color) != sizeof(CudaColor<unsigned char>)");

			// check cameras resolution
			if (hCamera->GetWidth() != hCudaCamera->width || 
				hCamera->GetHeight() != hCudaCamera->height) continue;
			if (hCamera->GetMaxWidth() != hCudaCamera->max_width || 
				hCamera->GetMaxHeight() != hCudaCamera->max_height) continue;

			size_t chunkSize = hCudaCamera->hostPinnedMemory.GetSize() / (sizeof(*hCudaCamera->final_image[m_update_ix]));
			if (chunkSize < 16u) ThrowException(L"Not enough host pinned memory for async image copy");

			size_t nPixels = hCamera->GetWidth() * hCamera->GetHeight();
			for (size_t startIndex = 0; startIndex < nPixels; startIndex += chunkSize)
			{
				if (startIndex + chunkSize > nPixels) chunkSize = nPixels - startIndex;

				// copy final image data from hCudaCamera to hCudaPixels on pinned memory
				CudaColor<unsigned char>* hCudaPixels = 
					(CudaColor<unsigned char>*)CudaCamera::hostPinnedMemory.GetPointerToMemory();
				CudaErrorCheck(cudaMemcpyAsync(hCudaPixels, hCudaCamera->final_image[m_update_ix] + startIndex, 
					chunkSize * sizeof(*hCudaPixels), 
					cudaMemcpyKind::cudaMemcpyDeviceToHost, *mirror_stream));
				CudaErrorCheck(cudaStreamSynchronize(*mirror_stream));

				// copy final image data from hostCudaPixels on pinned memory to hostCamera
				memcpy(hCamera->GetBitmap().GetMapAddress() + startIndex, hCudaPixels, 
					chunkSize * sizeof(*hCudaPixels));
			}
		}
	}

	void CudaEngine::LaunchFunction()
	{
		// TODO: change to m_render_ix when asynchronous //
		// TODO: change to m_render_stream when asynchronous //

		for (size_t i = 0; i < m_launch_configs[m_update_ix].size(); ++i)
		{
			LaunchConfiguration& configuration = m_launch_configs[m_update_ix][i];

			CudaKernel::Kernel
				<<<
				configuration.GetGrid(),
				configuration.GetThreadBlock(),
				configuration.GetSharedMemorySize(),
				m_mirror_stream
				>>>
				(mp_dCudaWorld);

			CudaErrorCheck(cudaStreamSynchronize(m_mirror_stream));
			CudaErrorCheck(cudaGetLastError());
		}
	}

}