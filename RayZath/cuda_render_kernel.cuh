#ifndef CUDA_ENGINE_KERNEL_CUH
#define CUDA_ENGINE_KERNEL_CUH

#include "cuda_world.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath::Cuda::Kernel
{
	// ~~~~~~~~ Memory Management ~~~~~~~~
	__host__ void CopyToConstantMemory(
		const ConstantKernel* hCudaConstantKernel,
		const uint32_t& update_idx,
		cudaStream_t& stream);


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
		TracingPath& tracing_path,
		RayIntersection& intersection,
		RNG& rng);
	__device__ void RenderCumulativePass(
		const World& World,
		Camera& camera,
		TracingPath& tracing_path,
		RayIntersection& intersection,
		RNG& rng);

	__device__ vec3f TraceRay(
		const World& World,
		TracingPath& tracing_path,
		RayIntersection& intersection,
		RNG& rng);


	__device__ ColorF PointLightSampling(
		const World& world,
		const RayIntersection& intersection,
		const vec3f& vS,
		const float vSw,
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
}

#endif // !CUDA_ENGINE_KERNEL_CUH