#ifndef CUDA_PREPROCESS_KERNEL_CUH
#define CUDA_PREPROCESS_KERNEL_CUH

#include "cuda_world.cuh"

namespace RayZath::Cuda::Kernel
{
	// cumulative samples management
	__global__ void DepthBufferReset(
		World* const world,
		const int camera_id);
	__global__ void CudaCameraUpdateSamplesNumber(
		World* const world,
		const int camera_id,
		bool reset_flag);
}

#endif