#include "cuda_engine.cuh"

namespace RayZath
{
	CudaEngine::CudaEngine()
		: mp_dCudaWorld(nullptr)
		, m_hpm_CudaWorld(sizeof(CudaWorld))
	{
		cudaSetDevice(0);

		// create streams
		CudaErrorCheck(cudaStreamCreate(&m_mirror_stream));

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
	}


	void CudaEngine::RenderWorld(World& hWorld)
	{
		ReconstructCudaWorld(mp_dCudaWorld, hWorld, &m_mirror_stream);

		CudaKernel::Kernel<<<500u, 256u>>>(mp_dCudaWorld);

		CudaErrorCheck(cudaDeviceSynchronize());
		CudaErrorCheck(cudaGetLastError());

		TransferResultsToHost(mp_dCudaWorld, hWorld, &m_mirror_stream);
		/*CudaWorld* hCudaWorld = (CudaWorld*)malloc(sizeof(CudaWorld));
		CudaErrorCheck(cudaMemcpy(
			hCudaWorld, mp_dCudaWorld,
			sizeof(CudaWorld),
			cudaMemcpyKind::cudaMemcpyDeviceToHost));

		CudaCamera* hCudaCamera = (CudaCamera*)malloc(sizeof(CudaCamera));
		CudaErrorCheck(cudaMemcpy(
			hCudaCamera, &hCudaWorld->cameras[0],
			sizeof(CudaCamera),
			cudaMemcpyKind::cudaMemcpyDeviceToHost));

		float x = hCudaCamera->position.x;


		free(hCudaCamera);
		free(hCudaWorld);*/
		//CudaKernel::CallKernel();
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

			// [>] Get CudaCamera class from hostCudaWorld
			CudaCamera* hCudaCamera = (CudaCamera*)CudaWorld::m_hpm.GetPointerToMemory();
			if (CudaWorld::m_hpm.GetSize() < sizeof(*hCudaCamera))
				throw Exception(__FILE__, __LINE__, L"insufficient host pinned memory for CudaCamera");
			CudaErrorCheck(cudaMemcpyAsync(
				hCudaCamera, &hCudaWorld->cameras[i], 
				sizeof(CudaCamera), 
				cudaMemcpyKind::cudaMemcpyDeviceToHost, *mirror_stream));
			CudaErrorCheck(cudaStreamSynchronize(*mirror_stream));

			if (!hCudaCamera->Exist()) continue;

			hCamera->m_samples_count = hCudaCamera->samples_count;

			// [>] Asynchronous copying
			static_assert(
				sizeof(*hCamera->GetBitmap().GetMapAddress()) == 
/* change index */				sizeof(*hCudaCamera->final_image[0]), 
				"sizeof(Graphics::Color) != sizeof(CudaColor<unsigned char>)");

			// check cameras resolution
			if (hCamera->GetWidth() != hCudaCamera->width || 
				hCamera->GetHeight() != hCudaCamera->height) continue;
			if (hCamera->GetMaxWidth() != hCudaCamera->max_width || 
				hCamera->GetMaxHeight() != hCudaCamera->max_height) continue;

/* change index */			size_t chunkSize = hCudaCamera->hostPinnedMemory.GetSize() / (sizeof(*hCudaCamera->final_image[0]));
			if (chunkSize == 16u)
				throw Exception(__FILE__, __LINE__, 
					L"Not enough host pinned memory for async image copy");

			size_t nPixels = hCamera->GetWidth() * hCamera->GetHeight();
			for (size_t startIndex = 0; startIndex < nPixels; startIndex += chunkSize)
			{
				if (startIndex + chunkSize > nPixels) chunkSize = nPixels - startIndex;

				// copy final image data from hostCudaCamera to hostCudaPixels on pinned memory
				CudaColor<unsigned char>* hCudaPixels = (CudaColor<unsigned char>*)CudaCamera::hostPinnedMemory.GetPointerToMemory();
/* change index */				CudaErrorCheck(cudaMemcpyAsync(hCudaPixels, hCudaCamera->final_image[0] + startIndex, 
					chunkSize * sizeof(*hCudaPixels), 
					cudaMemcpyKind::cudaMemcpyDeviceToHost, *mirror_stream));
				CudaErrorCheck(cudaStreamSynchronize(*mirror_stream));

				// copy final image data from hostCudaPixels on pinned memory to hostCamera
				memcpy(hCamera->GetBitmap().GetMapAddress() + startIndex, hCudaPixels, 
					chunkSize * sizeof(*hCudaPixels));
			}
		}
	}
}