#ifndef CUDA_ENGINE_KERNEL_CUH
#define CUDA_ENGINE_KERNEL_CUH

#include "cuda_world.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		namespace CudaKernel
		{
			// ~~~~~~~~ Memory Management ~~~~~~~~
			__host__ void CopyToConstantMemory(
				const CudaConstantKernel* hCudaConstantKernel,
				const uint32_t& update_idx,
				cudaStream_t& stream);


			// ~~~~~~~~ Rendering Functions ~~~~~~~~
			__global__ void GenerateCameraRay(
				CudaGlobalKernel* const global_kernel,
				CudaWorld* const world,
				const int camera_id);

			__device__ void Render(
				ThreadData& thread,
				const CudaWorld& World,
				CudaCamera& camera,
				TracingPath& tracing_path,
				RayIntersection& intersection);

			__device__ void TraceRay(
				ThreadData& thread,
				const CudaWorld& World,
				TracingPath& tracing_path,
				RayIntersection& intersection,
				ColorF& color_mask);

			__device__ Color<float> DirectSampling(
				ThreadData& thread,
				const CudaWorld& world,
				RayIntersection& intersection);
		}
	}
}

#endif // !CUDA_ENGINE_KERNEL_CUH