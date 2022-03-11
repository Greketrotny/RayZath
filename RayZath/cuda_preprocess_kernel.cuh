#ifndef CUDA_PREPROCESS_KERNEL_CUH
#define CUDA_PREPROCESS_KERNEL_CUH

#include "cuda_kernel_data.cuh"
#include "cuda_world.cuh"

namespace RayZath::Cuda::Kernel
{
	__global__ void PassReset(
		World* const world,
		const uint32_t camera_idx);

	__global__ void GenerateCameraRay(
		GlobalKernel* const global_kernel,
		World* const world,
		const uint32_t camera_idx);
}

#endif