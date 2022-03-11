#ifndef CUDA_POSTPROCESS_KERNEL_CUH
#define CUDA_POSTPROCESS_KERNEL_CUH

#include "cuda_kernel_data.cuh"
#include "cuda_render_parts.cuh"
#include "cuda_world.cuh"

namespace RayZath::Cuda::Kernel
{
	__global__ void SpacialReprojection(
		World* const world,
		const uint32_t camera_idx);

	__global__ void FirstToneMap(
		World* const world,
		const uint32_t camera_idx);
	__global__ void ToneMap(
		World* const world,
		const uint32_t camera_idx);

	__global__ void PassUpdate(
		World* const world,
		const uint32_t camera_idx);


	/*__global__ void IrradianceReduction(
		GlobalKernel* const global_kernel,
		World* const world,
		const int camera_id);*/
}

#endif // !CUDA_POSTPROCESS_KERNEL_CUH