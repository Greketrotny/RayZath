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

	__device__ void TraceRay(
		const World& World,
		TracingPath& tracing_path,
		RayIntersection& intersection,
		RNG& rng);

	__device__ Color<float> DirectSampling(
		const World& world,
		RayIntersection& intersection,
		RNG& rng);
}

#endif // !CUDA_ENGINE_KERNEL_CUH