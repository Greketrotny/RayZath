#ifndef CUDA_POSTPROCESS_KERNEL_CUH
#define CUDA_POSTPROCESS_KERNEL_CUH

#include "cuda_kernel_data.cuh"
#include "cuda_render_parts.cuh"
#include "cuda_world.cuh"

namespace RayZath::Cuda::Kernel
{
	__global__ void spacialReprojection(
		World* const world,
		const uint32_t camera_idx);

	__global__ void firstToneMap(
		World* const world,
		const uint32_t camera_idx);
	__global__ void toneMap(
		World* const world,
		const uint32_t camera_idx);

	__global__ void passUpdate(
		World* const world,
		const uint32_t camera_idx);

	__global__ void rayCast(
		World* const world,
		const uint32_t camera_dix);


	/*__global__ void IrradianceReduction(
		GlobalKernel* const global_kernel,
		World* const world,
		const int camera_id);*/
}

#endif // !CUDA_POSTPROCESS_KERNEL_CUH