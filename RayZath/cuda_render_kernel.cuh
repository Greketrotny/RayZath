#ifndef CUDA_ENGINE_KERNEL_CUH
#define CUDA_ENGINE_KERNEL_CUH

#include "cuda_kernel_data.cuh" 
#include "cuda_world.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath::Cuda::Kernel
{
	__global__ void RenderFirstPass(
		GlobalKernel* const global_kernel,
		World* const world,
		const uint32_t camera_idx);
	__global__ void RenderCumulativePass(
		GlobalKernel* const global_kernel,
		World* const world,
		const uint32_t camera_idx);
	__global__ void SegmentUpdate(
		World* const world,
		const uint32_t camera_idx);


	__device__ vec3f TraceRay(
		ConstantKernel& ckernel,
		const World& World,
		TracingState& tracing_state,
		RayIntersection& intersection,
		RNG& rng);

	__device__ ColorF SpotLightSampling(
		ConstantKernel& ckernel,
		const World& world,
		const RayIntersection& intersection,
		const vec3f& vS,
		const float vSw,
		RNG& rng);
	__device__ ColorF DirectLightSampling(
		ConstantKernel& ckernel,
		const World& world,
		const RayIntersection& intersection,
		const vec3f& vS,
		const float vSw,
		RNG& rng);
	__device__ ColorF DirectIllumination(
		ConstantKernel& ckernel,
		const World& world,
		const RayIntersection& intersection,
		const vec3f& vS,
		RNG& rng);
}

#endif // !CUDA_ENGINE_KERNEL_CUH