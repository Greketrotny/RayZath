#ifndef CUDA_ENGINE_KERNEL_CUH
#define CUDA_ENGINE_KERNEL_CUH

#include "cuda_kernel_data.cuh" 
#include "cuda_world.cuh"

namespace RayZath::Cuda::Kernel
{
	__global__ void renderFirstPass(
		GlobalKernel* const global_kernel,
		World* const world,
		const uint32_t camera_idx);
	__global__ void renderCumulativePass(
		GlobalKernel* const global_kernel,
		World* const world,
		const uint32_t camera_idx);
	__global__ void segmentUpdate(
		World* const world,
		const uint32_t camera_idx);
	__global__ void rayCast(
		World* const world,
		const uint32_t camera_dix);


	__device__ TracingResult traceRay(
		ConstantKernel& ckernel,
		const World& World,
		TracingState& tracing_state,
		SceneRay& ray,
		RNG& rng);

	__device__ ColorF spotLightSampling(
		ConstantKernel& ckernel,
		const World& world,
		const SceneRay& ray,
		const TracingResult& result,
		const SurfaceProperties& surface,
		const float vS_pdf,
		RNG& rng);
	__device__ ColorF directLightSampling(
		ConstantKernel& ckernel,
		const World& world,
		const SceneRay& ray,
		const TracingResult& result,
		const SurfaceProperties& surface,
		const float vS_pdf,
		RNG& rng);
	__device__ ColorF directIllumination(
		ConstantKernel& ckernel,
		const World& world,
		const SceneRay& ray,
		const TracingResult& result,
		const SurfaceProperties& surface,
		RNG& rng);
}

#endif // !CUDA_ENGINE_KERNEL_CUH
