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

		CudaKernel::Kernel<<<1u, 1u>>>(mp_dCudaWorld);

		CudaErrorCheck(cudaDeviceSynchronize());
		CudaErrorCheck(cudaGetLastError());

		CudaWorld* hCudaWorld = (CudaWorld*)malloc(sizeof(CudaWorld));
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
		free(hCudaWorld);
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
	}
}