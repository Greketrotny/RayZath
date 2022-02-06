#ifndef CUDA_ENGINE_KERNEL_CUH
#define CUDA_ENGINE_KERNEL_CUH

#include "cuda_kernel_data.cuh" 
#include "cuda_world.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath::Cuda::Kernel
{
	// ~~~~~~~~ Rendering Functions ~~~~~~~~
	__global__ void LaunchFirstPass(
		GlobalKernel* const global_kernel,
		World* const world,
		const int camera_id);
	__global__ void LaunchCumulativePass(
		GlobalKernel* const global_kernel,
		World* const world,
		const int camera_id);

	__device__ void RenderFirstPass(
		FullThread& thread,
		const World& World,
		Camera& camera,
		TracingState& tracing_state,
		RayIntersection& intersection,
		RNG& rng);
	__device__ void RenderCumulativePass(
		const World& World,
		Camera& camera,
		TracingState& tracing_state,
		RayIntersection& intersection,
		RNG& rng);

	__device__ vec3f TraceRay(
		const World& World,
		TracingState& tracing_state,
		RayIntersection& intersection,
		RNG& rng);


	__device__ ColorF SpotLightSampling(
		const World& world,
		const RayIntersection& intersection,
		const vec3f& vS,
		const float vSw,
		RNG& rng);
	__device__ ColorF DirectLightSampling(
		const World& world,
		const RayIntersection& intersection,
		const vec3f& vS,
		const float vSw,
		RNG& rng);
	__device__ ColorF DirectIllumination(
		const World& world,
		const RayIntersection& intersection,
		const vec3f& vS,
		RNG& rng);


	__global__ void ContRenderFirstPass(
		GlobalKernel* const global_kernel,
		World* const world,
		const int camera_idx);
	__global__ void ContRenderCumulativePass(
		GlobalKernel* const global_kernel,
		World* const world,
		const int camera_idx);
}

#endif // !CUDA_ENGINE_KERNEL_CUH